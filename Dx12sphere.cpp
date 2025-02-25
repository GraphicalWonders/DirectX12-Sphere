#include <windows.h>
#include <wrl.h>
#include <vector>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include "d3dx12.h"
#include <DirectXMath.h>
#include <iostream>
#include <sdl.h>
#include <sdl_syswm.h>
#include <cassert>
#include <cmath>

using namespace Microsoft::WRL;
using namespace DirectX;

static const int render_width = 800;
static const int render_height = 600;
static const UINT frame_count = 2;
static const UINT compute_buffer_size = 1024;
static const int shadow_map_width = 1024;
static const int shadow_map_height = 1024;

// wait for gpu to finish
void wait_for_gpu(ComPtr<ID3D12CommandQueue> cmd_queue,
    ComPtr<ID3D12Fence> fence_obj,
    HANDLE fence_event,
    UINT64& fence_val)
{
    fence_val++;
    cmd_queue->Signal(fence_obj.Get(), fence_val);
    if (fence_obj->GetCompletedValue() < fence_val)
    {
        fence_obj->SetEventOnCompletion(fence_val, fence_event);
        WaitForSingleObject(fence_event, INFINITE);
    }
}

struct scene_constants {
    XMMATRIX model_matrix;
    XMMATRIX view_matrix;
    XMMATRIX proj_matrix;
    XMMATRIX light_view_proj; // for shadow mapping
    XMFLOAT4 camera_position;
    XMFLOAT4 light_position;
    XMFLOAT4 ambient_color;
    XMFLOAT4 diffuse_color;
    XMFLOAT4 specular_color;
    float    specular_power;
    float    pad[3];
    XMFLOAT4 light2_position;
    XMFLOAT4 ambient_color2;
    XMFLOAT4 diffuse_color2;
    XMFLOAT4 specular_color2;
    float    specular_power2;
    float    pad2[3];
};

struct simple_vertex {
    XMFLOAT3 position;
    XMFLOAT3 normal;
};

ComPtr<ID3D12Resource>      compute_buffer;
ComPtr<ID3D12DescriptorHeap> compute_uav_heap;
ComPtr<ID3D12RootSignature>  compute_root_signature;
ComPtr<ID3D12PipelineState>  compute_pso;
ComPtr<ID3D12CommandAllocator> compute_cmd_allocator;
ComPtr<ID3D12GraphicsCommandList> compute_cmd_list;

void initialize_compute_shader(ID3D12Device* device)
{
    // create uav heap (shader visible)
    D3D12_DESCRIPTOR_HEAP_DESC uav_heap_desc = {};
    uav_heap_desc.NumDescriptors = 1;
    uav_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    uav_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    device->CreateDescriptorHeap(&uav_heap_desc, IID_PPV_ARGS(&compute_uav_heap));

    // create compute buffer and its uav
    UINT buffer_size = compute_buffer_size * sizeof(UINT);
    CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(buffer_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buffer_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&compute_buffer));

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = compute_buffer_size;
    uav_desc.Buffer.StructureByteStride = sizeof(UINT);
    uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    device->CreateUnorderedAccessView(compute_buffer.Get(), nullptr, &uav_desc,
        compute_uav_heap->GetCPUDescriptorHandleForHeapStart());

    // simple compute shader source (doubles thread id)
    const char* compute_shader_src = R"(
        RWStructuredBuffer<uint> resultBuffer : register(u0);
        [numthreads(64,1,1)]
        void mainCS(uint3 DTid : SV_DispatchThreadID)
        {
            resultBuffer[DTid.x] = DTid.x * 2;
        }
    )";

    ComPtr<ID3DBlob> cs_blob, cs_error;
    HRESULT hr = D3DCompile(compute_shader_src, strlen(compute_shader_src),
        nullptr, nullptr, nullptr, "mainCS", "cs_5_0", 0, 0,
        &cs_blob, &cs_error);
    if (FAILED(hr))
    {
        if (cs_error)
            std::cerr << (char*)cs_error->GetBufferPointer() << std::endl;
        exit(-1);
    }

    // setup descriptor range and root parameter
    CD3DX12_DESCRIPTOR_RANGE range = {};
    range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    CD3DX12_ROOT_PARAMETER root_param = {};
    root_param.InitAsDescriptorTable(1, &range);

    // create and serialize root signature
    CD3DX12_ROOT_SIGNATURE_DESC compute_root_sig_desc = {};
    compute_root_sig_desc.Init(1, &root_param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> serialized_root_sig, error_blob;
    hr = D3D12SerializeRootSignature(&compute_root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized_root_sig, &error_blob);
    if (FAILED(hr))
    {
        if (error_blob)
            std::cerr << (char*)error_blob->GetBufferPointer() << std::endl;
        exit(-1);
    }
    device->CreateRootSignature(0, serialized_root_sig->GetBufferPointer(),
        serialized_root_sig->GetBufferSize(),
        IID_PPV_ARGS(&compute_root_signature));

    // create compute pipeline state
    D3D12_COMPUTE_PIPELINE_STATE_DESC compute_pso_desc = {};
    compute_pso_desc.pRootSignature = compute_root_signature.Get();
    compute_pso_desc.CS = { cs_blob->GetBufferPointer(), cs_blob->GetBufferSize() };
    hr = device->CreateComputePipelineState(&compute_pso_desc, IID_PPV_ARGS(&compute_pso));
    if (FAILED(hr))
    {
        std::cerr << "failed to create compute pipeline state." << std::endl;
        exit(-1);
    }

    // create command allocator and list for compute
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&compute_cmd_allocator));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        compute_cmd_allocator.Get(), compute_pso.Get(),
        IID_PPV_ARGS(&compute_cmd_list));
    compute_cmd_list->Close();
}

int main(int, char**)
{
    // sdl init and window creation
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << SDL_GetError() << "\n";
        return -1;
    }
    SDL_Window* window = SDL_CreateWindow("dx12 sphere",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        render_width, render_height,
        SDL_WINDOW_SHOWN);
    if (!window)
    {
        std::cerr << SDL_GetError() << "\n";
        SDL_Quit();
        return -1;
    }
    SDL_SysWMinfo wm_info;
    SDL_VERSION(&wm_info.version);
    if (!SDL_GetWindowWMInfo(window, &wm_info))
    {
        std::cerr << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    HWND hwnd = wm_info.info.win.window;

    // create dx12 factory and device
    ComPtr<IDXGIFactory4> factory;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    ComPtr<ID3D12Device> device;
    D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));

    // command queue creation
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> command_queue;
    device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&command_queue));

    // check msaa support
    UINT msaa_samples = 8;
    UINT msaa_quality = 0;
    {
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaa_info = {};
        msaa_info.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        msaa_info.SampleCount = msaa_samples;
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
            &msaa_info, sizeof(msaa_info))) ||
            msaa_info.NumQualityLevels == 0)
        {
            msaa_samples = 1;
            msaa_quality = 0;
            std::cerr << "msaa not supported, using 1x.\n";
        }
        else {
            msaa_quality = msaa_info.NumQualityLevels - 1;
        }
    }

    // swap chain creation
    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.BufferCount = frame_count;
    scd.Width = render_width;
    scd.Height = render_height;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;
    scd.SampleDesc.Quality = 0;
    ComPtr<IDXGISwapChain1> swap_chain1;
    factory->CreateSwapChainForHwnd(command_queue.Get(), hwnd, &scd,
        nullptr, nullptr, &swap_chain1);
    ComPtr<IDXGISwapChain3> swap_chain;
    swap_chain1.As(&swap_chain);
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    UINT frame_index = swap_chain->GetCurrentBackBufferIndex();

    // create render target view heap and back buffers
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc = {};
    rtv_heap_desc.NumDescriptors = frame_count;
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap));
    UINT rtv_desc_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    ComPtr<ID3D12Resource> back_buffers[frame_count];
    {
        CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(rtv_heap->GetCPUDescriptorHandleForHeapStart());
        for (UINT i = 0; i < frame_count; i++)
        {
            swap_chain->GetBuffer(i, IID_PPV_ARGS(&back_buffers[i]));
            device->CreateRenderTargetView(back_buffers[i].Get(), nullptr, rtv_handle);
            rtv_handle.Offset(1, rtv_desc_size);
        }
    }

    // create msaa target heap and resources
    D3D12_DESCRIPTOR_HEAP_DESC msaa_heap_desc = {};
    msaa_heap_desc.NumDescriptors = frame_count;
    msaa_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    msaa_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ComPtr<ID3D12DescriptorHeap> msaa_heap;
    device->CreateDescriptorHeap(&msaa_heap_desc, IID_PPV_ARGS(&msaa_heap));
    UINT msaa_desc_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    ComPtr<ID3D12Resource> msaa_targets[frame_count];
    {
        CD3DX12_CPU_DESCRIPTOR_HANDLE msaa_handle(msaa_heap->GetCPUDescriptorHandleForHeapStart());
        for (UINT i = 0; i < frame_count; i++)
        {
            D3D12_RESOURCE_DESC desc = {};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = render_width;
            desc.Height = render_height;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count = msaa_samples;
            desc.SampleDesc.Quality = msaa_quality;
            desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            D3D12_CLEAR_VALUE clear_val = {};
            clear_val.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            clear_val.Color[0] = 1.f; clear_val.Color[1] = 1.f;
            clear_val.Color[2] = 1.f; clear_val.Color[3] = 1.f;
            CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_DEFAULT);
            device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET, &clear_val,
                IID_PPV_ARGS(&msaa_targets[i]));
            D3D12_RENDER_TARGET_VIEW_DESC rtv_desc = {};
            rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            device->CreateRenderTargetView(msaa_targets[i].Get(), &rtv_desc, msaa_handle);
            msaa_handle.Offset(1, msaa_desc_size);
        }
    }

    // create depth stencil view heap and depth buffer
    D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc = {};
    dsv_heap_desc.NumDescriptors = 1;
    dsv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ComPtr<ID3D12DescriptorHeap> dsv_heap;
    device->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap));
    ComPtr<ID3D12Resource> depth_buffer;
    {
        D3D12_RESOURCE_DESC depth_desc = {};
        depth_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        depth_desc.Width = render_width;
        depth_desc.Height = render_height;
        depth_desc.DepthOrArraySize = 1;
        depth_desc.MipLevels = 1;
        depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
        depth_desc.SampleDesc.Count = msaa_samples;
        depth_desc.SampleDesc.Quality = msaa_quality;
        depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        depth_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        D3D12_CLEAR_VALUE depth_clear = {};
        depth_clear.Format = DXGI_FORMAT_D32_FLOAT;
        depth_clear.DepthStencil.Depth = 1.0f;
        depth_clear.DepthStencil.Stencil = 0;
        CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_DEFAULT);
        device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &depth_desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &depth_clear,
            IID_PPV_ARGS(&depth_buffer));
    }
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
        dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
        dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
        dsv_desc.Flags = D3D12_DSV_FLAG_NONE;
        device->CreateDepthStencilView(depth_buffer.Get(), &dsv_desc,
            dsv_heap->GetCPUDescriptorHandleForHeapStart());
    }

    // shadow map resource
    D3D12_RESOURCE_DESC shadow_desc = {};
    shadow_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    shadow_desc.Width = shadow_map_width;
    shadow_desc.Height = shadow_map_height;
    shadow_desc.DepthOrArraySize = 1;
    shadow_desc.MipLevels = 1;
    shadow_desc.Format = DXGI_FORMAT_D32_FLOAT;
    shadow_desc.SampleDesc.Count = 1;
    shadow_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CD3DX12_HEAP_PROPERTIES shadow_heap_props(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_CLEAR_VALUE shadow_clear = {};
    shadow_clear.Format = DXGI_FORMAT_D32_FLOAT;
    shadow_clear.DepthStencil.Depth = 1.0f;
    shadow_clear.DepthStencil.Stencil = 0;
    ComPtr<ID3D12Resource> shadow_map;
    device->CreateCommittedResource(&shadow_heap_props, D3D12_HEAP_FLAG_NONE, &shadow_desc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, &shadow_clear,
        IID_PPV_ARGS(&shadow_map));

    // create dsv for shadow map
    D3D12_DESCRIPTOR_HEAP_DESC shadow_dsv_heap_desc = {};
    shadow_dsv_heap_desc.NumDescriptors = 1;
    shadow_dsv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    shadow_dsv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ComPtr<ID3D12DescriptorHeap> shadow_dsv_heap;
    device->CreateDescriptorHeap(&shadow_dsv_heap_desc, IID_PPV_ARGS(&shadow_dsv_heap));
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC shadow_dsv_desc = {};
        shadow_dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
        shadow_dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        shadow_dsv_desc.Flags = D3D12_DSV_FLAG_NONE;
        device->CreateDepthStencilView(shadow_map.Get(), &shadow_dsv_desc,
            shadow_dsv_heap->GetCPUDescriptorHandleForHeapStart());
    }

    // create command allocator and list for graphics
    ComPtr<ID3D12CommandAllocator> cmd_allocator;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmd_allocator));
    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        cmd_allocator.Get(), nullptr,
        IID_PPV_ARGS(&cmd_list));
    cmd_list->Close();

    // create fence and event for gpu sync
    ComPtr<ID3D12Fence> fence_obj;
    UINT64 fence_val = 0;
    device->CreateFence(fence_val, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_obj));
    HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // initialize compute shader and its command list
    initialize_compute_shader(device.Get());
    ComPtr<ID3D12CommandAllocator> compute_cmd_allocator2;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&compute_cmd_allocator2));
    ComPtr<ID3D12GraphicsCommandList> compute_cmd_list2;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        compute_cmd_allocator2.Get(),
        compute_pso.Get(),
        IID_PPV_ARGS(&compute_cmd_list2));
    compute_cmd_list2->Close();

    // create compute readback buffer
    ComPtr<ID3D12Resource> compute_readback_buffer;
    {
        CD3DX12_HEAP_PROPERTIES readback_heap_props(D3D12_HEAP_TYPE_READBACK);
        CD3DX12_RESOURCE_DESC readback_buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(compute_buffer_size * sizeof(UINT));
        device->CreateCommittedResource(&readback_heap_props, D3D12_HEAP_FLAG_NONE,
            &readback_buffer_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&compute_readback_buffer));
    }

    // create sphere geometry
    std::vector<simple_vertex> sphere_vertices;
    std::vector<UINT> sphere_indices;
    {
        const UINT lat_bands = 40, lon_bands = 40;
        const float radius = 1.f;
        for (UINT lat = 0; lat <= lat_bands; lat++)
        {
            float theta = lat * XM_PI / lat_bands;
            float sin_theta = sinf(theta), cos_theta = cosf(theta);
            for (UINT lon = 0; lon <= lon_bands; lon++)
            {
                float phi = lon * 2.f * XM_PI / lon_bands;
                float sin_phi = sinf(phi), cos_phi = cosf(phi);
                float x = cos_phi * sin_theta;
                float y = cos_theta;
                float z = sin_phi * sin_theta;
                XMFLOAT3 pos(x * radius, y * radius, z * radius);
                XMFLOAT3 norm(x, y, z);
                XMVECTOR norm_vec = XMVector3Normalize(XMLoadFloat3(&norm));
                XMStoreFloat3(&norm, norm_vec);
                sphere_vertices.push_back({ pos, norm });
            }
        }
        for (UINT lat = 0; lat < lat_bands; lat++)
        {
            for (UINT lon = 0; lon < lon_bands; lon++)
            {
                UINT first = lat * (lon_bands + 1) + lon;
                UINT second = first + lon_bands + 1;
                sphere_indices.push_back(first);
                sphere_indices.push_back(second);
                sphere_indices.push_back(first + 1);
                sphere_indices.push_back(second);
                sphere_indices.push_back(second + 1);
                sphere_indices.push_back(first + 1);
            }
        }
    }
    UINT sphere_index_count = (UINT)sphere_indices.size();

    // create vertex buffer and view
    UINT vb_size = (UINT)(sphere_vertices.size() * sizeof(simple_vertex));
    ComPtr<ID3D12Resource> vertex_buffer;
    {
        CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(vb_size);
        device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&vertex_buffer));
        void* mapped_data = nullptr;
        vertex_buffer->Map(0, nullptr, &mapped_data);
        memcpy(mapped_data, sphere_vertices.data(), vb_size);
        vertex_buffer->Unmap(0, nullptr);
    }
    D3D12_VERTEX_BUFFER_VIEW vb_view = {};
    vb_view.BufferLocation = vertex_buffer->GetGPUVirtualAddress();
    vb_view.StrideInBytes = sizeof(simple_vertex);
    vb_view.SizeInBytes = vb_size;

    // create index buffer and view
    UINT ib_size = (UINT)(sphere_indices.size() * sizeof(UINT));
    ComPtr<ID3D12Resource> index_buffer;
    {
        CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(ib_size);
        device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&index_buffer));
        void* mapped_data = nullptr;
        index_buffer->Map(0, nullptr, &mapped_data);
        memcpy(mapped_data, sphere_indices.data(), ib_size);
        index_buffer->Unmap(0, nullptr);
    }
    D3D12_INDEX_BUFFER_VIEW ib_view = {};
    ib_view.BufferLocation = index_buffer->GetGPUVirtualAddress();
    ib_view.Format = DXGI_FORMAT_R32_UINT;
    ib_view.SizeInBytes = ib_size;

    // shader with noise and shadow mapping
    const char* shader_src = R"(
        cbuffer scene_cb : register(b0)
        {
            float4x4 model_matrix;
            float4x4 view_matrix;
            float4x4 proj_matrix;
            float4x4 light_view_proj;
            float4   camera_position;
            float4   light_position;
            float4   ambient_color;
            float4   diffuse_color;
            float4   specular_color;
            float    specular_power;
            float    pad[3];
            float4   light2_position;
            float4   ambient_color2;
            float4   diffuse_color2;
            float4   specular_color2;
            float    specular_power2;
            float    pad2[3];
        }

        cbuffer time_cb : register(b1)
        {
            float time;
        }

        struct vs_in
        {
            float3 pos : POSITION;
            float3 nor : NORMAL;
        };

        struct ps_in
        {
            float4 pos       : SV_POSITION;
            float3 world_pos : TEXCOORD0;
            float3 world_nor : TEXCOORD1;
            float4 light_space_pos : TEXCOORD2;
        };

        float noise(float2 p)
        {
            return frac(sin(dot(p, float2(12.9898,78.233))) * 43758.5453);
        }

        ps_in main_vs(vs_in input)
        {
            ps_in output;
            float4 wpos = mul(float4(input.pos, 1), model_matrix);
            output.world_pos = wpos.xyz;
            float4 wnor = mul(float4(input.nor, 0), model_matrix);
            output.world_nor = normalize(wnor.xyz);
            float4 vpos = mul(wpos, view_matrix);
            output.pos = mul(vpos, proj_matrix);
            output.light_space_pos = mul(wpos, light_view_proj);
            return output;
        }

        float4 main_ps(ps_in input) : SV_TARGET
        {
            float3 projCoords = input.light_space_pos.xyz / input.light_space_pos.w;
            projCoords = projCoords * 0.5f + 0.5f;
            float shadow = (projCoords.x > 0 && projCoords.x < 1 &&
                            projCoords.y > 0 && projCoords.y < 1) ? 0.7f : 1.0f;
            float n = noise(input.world_pos.xz * 0.1);
            float frequency = 10.0f;
            float speed = 2.0f;
            float amplitude = 0.1f;
            float rippleX = sin(input.world_pos.x * frequency + time * speed + n) * amplitude;
            float rippleZ = sin(input.world_pos.z * frequency + time * speed + n) * amplitude;
            float3 perturbed_n = normalize(input.world_nor + float3(rippleX, 0, rippleZ));
            float3 nrm = perturbed_n;
            float3 v = normalize(camera_position.xyz - input.world_pos);
            float ndotv = saturate(dot(nrm, v));
            float3 l = normalize(light_position.xyz - input.world_pos);
            float ndotl = saturate(dot(nrm, l));
            float3 h = normalize(l + v);
            float ndoth = saturate(dot(nrm, h));
            float vdoth = saturate(dot(v, h));
            float roughness = 0.03f;
            float alpha = roughness * roughness;
            float alpha2 = alpha * alpha;
            float denom = (ndoth * ndoth) * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (3.14159265 * denom * denom);
            float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
            float G_V = ndotv / (ndotv * (1.0 - k) + k);
            float G_L = ndotl / (ndotl * (1.0 - k) + k);
            float G = G_V * G_L;
            float3 F0 = specular_color.xyz;
            float3 F = F0 + (1.0 - F0) * pow(1.0 - vdoth, 5.0);
            float3 spec = (D * G * F) / (4.0 * ndotv * ndotl + 0.001);
            spec *= 3.0f;
            float3 diff = diffuse_color.xyz * ndotl;
            float3 amb = ambient_color.xyz;
            float occ = dot(nrm, v);
            float t_val = saturate(0.5 + 0.5 * occ);
            t_val = t_val * 0.7 + 0.3;
            amb *= t_val;
            float3 color = (amb + diff + spec) * shadow;
            color = pow(color, 1.0 / 2.2);
            return float4(color, 1.0f);
        }
    )";

    // compile shaders for main pass
    ComPtr<ID3DBlob> vs_blob, ps_blob, err_blob;
    HRESULT hr = D3DCompile(shader_src, strlen(shader_src), nullptr,
        nullptr, nullptr, "main_vs", "vs_5_0", 0, 0,
        &vs_blob, &err_blob);
    if (FAILED(hr))
    {
        if (err_blob)
            std::cerr << (char*)err_blob->GetBufferPointer() << std::endl;
        return -1;
    }
    hr = D3DCompile(shader_src, strlen(shader_src), nullptr,
        nullptr, nullptr, "main_ps", "ps_5_0", 0, 0,
        &ps_blob, &err_blob);
    if (FAILED(hr))
    {
        if (err_blob)
            std::cerr << (char*)err_blob->GetBufferPointer() << std::endl;
        return -1;
    }

    // create root signature for main pass (2 constant buffers)
    CD3DX12_ROOT_PARAMETER root_params[2];
    root_params[0].InitAsConstantBufferView(0);
    root_params[1].InitAsConstantBufferView(1);
    CD3DX12_ROOT_SIGNATURE_DESC root_sig_desc2;
    root_sig_desc2.Init(2, root_params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    ComPtr<ID3DBlob> root_sig_blob;
    D3D12SerializeRootSignature(&root_sig_desc2, D3D_ROOT_SIGNATURE_VERSION_1,
        &root_sig_blob, &err_blob);
    ComPtr<ID3D12RootSignature> root_signature;
    device->CreateRootSignature(0, root_sig_blob->GetBufferPointer(),
        root_sig_blob->GetBufferSize(),
        IID_PPV_ARGS(&root_signature));

    D3D12_INPUT_ELEMENT_DESC input_elems[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    // create graphics pipeline state for main pass
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.InputLayout = { input_elems, 2 };
    pso_desc.pRootSignature = root_signature.Get();
    pso_desc.VS = { vs_blob->GetBufferPointer(), vs_blob->GetBufferSize() };
    pso_desc.PS = { ps_blob->GetBufferPointer(), ps_blob->GetBufferSize() };
    pso_desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    pso_desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    pso_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    pso_desc.SampleMask = UINT_MAX;
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso_desc.SampleDesc.Count = msaa_samples;
    pso_desc.SampleDesc.Quality = msaa_quality;
    pso_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    ComPtr<ID3D12PipelineState> pipeline_state;
    device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pipeline_state));

    // shadow pass: vertex shader only
    const char* shadow_shader_src = R"(
        cbuffer scene_cb : register(b0)
        {
            float4x4 model_matrix;
            float4x4 light_view_proj;
        }
        struct vs_in
        {
            float3 pos : POSITION;
        };
        struct vs_out
        {
            float4 pos : SV_POSITION;
        };
        vs_out main_vs(vs_in input)
        {
            vs_out output;
            float4 world_pos = mul(float4(input.pos,1), model_matrix);
            output.pos = mul(world_pos, light_view_proj);
            return output;
        }
    )";
    ComPtr<ID3DBlob> shadow_vs_blob, shadow_err_blob;
    hr = D3DCompile(shadow_shader_src, strlen(shadow_shader_src), nullptr,
        nullptr, nullptr, "main_vs", "vs_5_0", 0, 0,
        &shadow_vs_blob, &shadow_err_blob);
    if (FAILED(hr))
    {
        if (shadow_err_blob)
            std::cerr << (char*)shadow_err_blob->GetBufferPointer() << std::endl;
        return -1;
    }
    // create root signature for shadow pass (1 constant buffer)
    CD3DX12_ROOT_PARAMETER shadow_root_param;
    shadow_root_param.InitAsConstantBufferView(0);
    CD3DX12_ROOT_SIGNATURE_DESC shadow_root_sig_desc;
    shadow_root_sig_desc.Init(1, &shadow_root_param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    ComPtr<ID3DBlob> shadow_root_sig_blob;
    D3D12SerializeRootSignature(&shadow_root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1,
        &shadow_root_sig_blob, &shadow_err_blob);
    ComPtr<ID3D12RootSignature> shadow_root_signature;
    device->CreateRootSignature(0, shadow_root_sig_blob->GetBufferPointer(),
        shadow_root_sig_blob->GetBufferSize(),
        IID_PPV_ARGS(&shadow_root_signature));
    D3D12_INPUT_ELEMENT_DESC shadow_input_elem[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC shadow_pso_desc = {};
    shadow_pso_desc.InputLayout = { shadow_input_elem, 1 };
    shadow_pso_desc.pRootSignature = shadow_root_signature.Get();
    shadow_pso_desc.VS = { shadow_vs_blob->GetBufferPointer(), shadow_vs_blob->GetBufferSize() };
    shadow_pso_desc.PS = { nullptr, 0 };
    shadow_pso_desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    shadow_pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    shadow_pso_desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    shadow_pso_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    shadow_pso_desc.SampleMask = UINT_MAX;
    shadow_pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    shadow_pso_desc.NumRenderTargets = 0;
    shadow_pso_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    shadow_pso_desc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> shadow_pipeline_state;
    device->CreateGraphicsPipelineState(&shadow_pso_desc, IID_PPV_ARGS(&shadow_pipeline_state));

    // create constant buffers
    const UINT cb_size = 256;
    ComPtr<ID3D12Resource> cb_resource;
    {
        CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(cb_size);
        device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE,
            &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&cb_resource));
    }
    UINT8* cb_ptr = nullptr;
    cb_resource->Map(0, nullptr, reinterpret_cast<void**>(&cb_ptr));

    ComPtr<ID3D12Resource> time_cb_resource;
    {
        CD3DX12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(16);
        device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE,
            &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&time_cb_resource));
    }
    UINT8* time_cb_ptr = nullptr;
    time_cb_resource->Map(0, nullptr, reinterpret_cast<void**>(&time_cb_ptr));

    // setup matrices and camera
    XMMATRIX view_m = XMMatrixLookAtLH(XMVectorSet(0.f, 0.f, -10.f, 0.f),
        XMVectorSet(0.f, 0.f, 0.f, 0.f),
        XMVectorSet(0.f, 1.f, 0.f, 0.f));
    XMMATRIX proj_m = XMMatrixPerspectiveFovLH(XM_PIDIV4,
        float(render_width) / float(render_height),
        0.1f, 100.f);
    XMMATRIX model_m = XMMatrixIdentity();
    model_m = XMMatrixTranspose(model_m);
    view_m = XMMatrixTranspose(view_m);
    proj_m = XMMatrixTranspose(proj_m);

    // light view-projection for shadow mapping (orthographic)
    XMVECTOR lightPos = XMVectorSet(5.f, 10.f, 5.f, 1.f);
    XMVECTOR lightTarget = XMVectorSet(0.f, 0.f, 0.f, 1.f);
    XMVECTOR upDir = XMVectorSet(0.f, 1.f, 0.f, 0.f);
    XMMATRIX lightView = XMMatrixLookAtLH(lightPos, lightTarget, upDir);
    XMMATRIX lightProj = XMMatrixOrthographicLH(20.f, 20.f, 0.1f, 50.f);
    XMMATRIX light_view_proj = XMMatrixTranspose(lightView * lightProj);

    scene_constants constants = {};
    constants.model_matrix = model_m;
    constants.view_matrix = view_m;
    constants.proj_matrix = proj_m;
    constants.light_view_proj = light_view_proj;
    constants.light_position = XMFLOAT4(0.f, 5.f, 0.f, 1.f);
    constants.light2_position = XMFLOAT4(0.f, 5.f, 0.f, 1.f);
    constants.camera_position = XMFLOAT4(0.f, 0.f, -10.f, 1.f);
    constants.ambient_color = XMFLOAT4(0.5f, 0.5f, 0.5f, 1.f);
    constants.diffuse_color = XMFLOAT4(1.f, 1.f, 1.f, 1.f);
    constants.specular_color = XMFLOAT4(1.f, 1.f, 1.f, 1.f);
    constants.specular_power = 32.f;
    constants.ambient_color2 = XMFLOAT4(0.5f, 0.5f, 0.5f, 1.f);
    constants.diffuse_color2 = XMFLOAT4(1.f, 1.f, 1.f, 1.f);
    constants.specular_color2 = XMFLOAT4(1.f, 1.f, 1.f, 1.f);
    constants.specular_power2 = 32.f;

    bool running = true;
    while (running)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_QUIT)
            {
                running = false;
                break;
            }
        }
        // update animated values
        float time = SDL_GetTicks() * 0.001f;
        float r_color = 0.5f * (sinf(time) + 1.0f);
        float g_color = 0.5f * (sinf(time + 2.094f) + 1.0f);
        float b_color = 0.5f * (sinf(time + 4.188f) + 1.0f);
        float r2_color = 0.5f * (sinf(time + 1.0f) + 1.0f);
        float g2_color = 0.5f * (sinf(time + 3.0f) + 1.0f);
        float b2_color = 0.5f * (sinf(time + 5.0f) + 1.0f);
        constants.light_position = XMFLOAT4(5.0f * cosf(time), 5.f, 5.0f * sinf(time), 1.f);
        constants.light2_position = XMFLOAT4(5.0f * cosf(-time), 5.f, 5.0f * sinf(-time), 1.f);
        constants.diffuse_color = XMFLOAT4(r_color, g_color, b_color, 1.f);
        constants.specular_color = XMFLOAT4(r_color, g_color, b_color, 1.f);
        constants.ambient_color = XMFLOAT4(0.5f * r_color, 0.5f * g_color, 0.5f * b_color, 1.f);
        constants.diffuse_color2 = XMFLOAT4(r2_color, g2_color, b2_color, 1.f);
        constants.specular_color2 = XMFLOAT4(r2_color, g2_color, b2_color, 1.f);
        constants.ambient_color2 = XMFLOAT4(0.5f * r2_color, 0.5f * g2_color, 0.5f * b2_color, 1.f);

        // update sphere model with unpredictable motion
        float randomX = 2.0f * sinf(time) + 1.5f * sinf(1.73f * time) + 1.0f * sinf(0.73f * time);
        float randomY = 2.0f * sinf(time + 1.0f) + 1.5f * sinf(1.73f * time + 0.5f) + 1.0f * sinf(0.73f * time + 1.2f);
        float randomZ = 2.0f * sinf(time + 2.0f) + 1.5f * sinf(1.73f * time + 1.0f) + 1.0f * sinf(0.73f * time + 0.3f);
        XMMATRIX translation = XMMatrixTranslation(randomX, randomY, randomZ);
        XMMATRIX rotation = XMMatrixRotationY(time);
        constants.model_matrix = XMMatrixTranspose(rotation * translation); 

        constants.light_view_proj = light_view_proj;
        memcpy(cb_ptr, &constants, sizeof(constants));
        memcpy(time_cb_ptr, &time, sizeof(float));

        // shadow map pass
        cmd_allocator->Reset();
        cmd_list->Reset(cmd_allocator.Get(), shadow_pipeline_state.Get());
        {
            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                shadow_map.Get(),
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE);
            cmd_list->ResourceBarrier(1, &barrier);
        }
        D3D12_VIEWPORT shadow_viewport = { 0.f, 0.f, (float)shadow_map_width, (float)shadow_map_height, 0.f, 1.f };
        D3D12_RECT shadow_scissor = { 0, 0, shadow_map_width, shadow_map_height };
        cmd_list->RSSetViewports(1, &shadow_viewport);
        cmd_list->RSSetScissorRects(1, &shadow_scissor);
        D3D12_CPU_DESCRIPTOR_HANDLE shadowDSVHandle = shadow_dsv_heap->GetCPUDescriptorHandleForHeapStart();
        cmd_list->OMSetRenderTargets(0, nullptr, FALSE, &shadowDSVHandle);
        cmd_list->ClearDepthStencilView(shadow_dsv_heap->GetCPUDescriptorHandleForHeapStart(),
            D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        cmd_list->SetGraphicsRootSignature(shadow_root_signature.Get());
        cmd_list->SetGraphicsRootConstantBufferView(0, cb_resource->GetGPUVirtualAddress());
        cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd_list->IASetVertexBuffers(0, 1, &vb_view);
        cmd_list->IASetIndexBuffer(&ib_view);
        cmd_list->DrawIndexedInstanced(sphere_index_count, 1, 0, 0, 0);
        {
            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                shadow_map.Get(),
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            cmd_list->ResourceBarrier(1, &barrier);
        }
        cmd_list->Close();
        ID3D12CommandList* shadow_lists[] = { cmd_list.Get() };
        command_queue->ExecuteCommandLists(_countof(shadow_lists), shadow_lists);
        wait_for_gpu(command_queue, fence_obj, fence_event, fence_val);

        // main render pass
        cmd_allocator->Reset();
        cmd_list->Reset(cmd_allocator.Get(), pipeline_state.Get());
        {
            CD3DX12_CPU_DESCRIPTOR_HANDLE msaa_handle(msaa_heap->GetCPUDescriptorHandleForHeapStart(), frame_index, msaa_desc_size);
            CD3DX12_CPU_DESCRIPTOR_HANDLE dsv_handle(dsv_heap->GetCPUDescriptorHandleForHeapStart());
            cmd_list->OMSetRenderTargets(1, &msaa_handle, FALSE, &dsv_handle);
            float clear_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            cmd_list->ClearRenderTargetView(msaa_handle, clear_color, 0, nullptr);
            cmd_list->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        }
        cmd_list->SetGraphicsRootSignature(root_signature.Get());
        cmd_list->SetPipelineState(pipeline_state.Get());
        D3D12_VIEWPORT viewport = { 0.f, 0.f, (float)render_width, (float)render_height, 0.f, 1.f };
        D3D12_RECT scissor_rect = { 0, 0, render_width, render_height };
        cmd_list->RSSetViewports(1, &viewport);
        cmd_list->RSSetScissorRects(1, &scissor_rect);
        cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd_list->IASetVertexBuffers(0, 1, &vb_view);
        cmd_list->IASetIndexBuffer(&ib_view);
        cmd_list->SetGraphicsRootConstantBufferView(0, cb_resource->GetGPUVirtualAddress());
        cmd_list->SetGraphicsRootConstantBufferView(1, time_cb_resource->GetGPUVirtualAddress());
        cmd_list->DrawIndexedInstanced(sphere_index_count, 1, 0, 0, 0);
        {
            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                msaa_targets[frame_index].Get(),
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            cmd_list->ResourceBarrier(1, &barrier);
        }
        {
            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                back_buffers[frame_index].Get(),
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RESOLVE_DEST);
            cmd_list->ResourceBarrier(1, &barrier);
        }
        cmd_list->ResolveSubresource(back_buffers[frame_index].Get(), 0,
            msaa_targets[frame_index].Get(), 0,
            DXGI_FORMAT_R8G8B8A8_UNORM);
        {
            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                back_buffers[frame_index].Get(),
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
                D3D12_RESOURCE_STATE_PRESENT);
            cmd_list->ResourceBarrier(1, &barrier);
        }
        {
            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                msaa_targets[frame_index].Get(),
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET);
            cmd_list->ResourceBarrier(1, &barrier);
        }
        cmd_list->Close();
        ID3D12CommandList* graphics_lists[] = { cmd_list.Get() };
        command_queue->ExecuteCommandLists(_countof(graphics_lists), graphics_lists);

        // compute shader pass
        compute_cmd_allocator2->Reset();
        compute_cmd_list2->Reset(compute_cmd_allocator2.Get(), compute_pso.Get());
        ID3D12DescriptorHeap* ppHeaps[] = { compute_uav_heap.Get() };
        compute_cmd_list2->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
        compute_cmd_list2->SetComputeRootSignature(compute_root_signature.Get());
        compute_cmd_list2->SetComputeRootDescriptorTable(0, compute_uav_heap->GetGPUDescriptorHandleForHeapStart());
        compute_cmd_list2->Dispatch(16, 1, 1);
        compute_cmd_list2->Close();
        ID3D12CommandList* compute_lists[] = { compute_cmd_list2.Get() };
        command_queue->ExecuteCommandLists(_countof(compute_lists), compute_lists);

        static bool check_done = false;
        if (!check_done)
        {
            compute_cmd_allocator2->Reset();
            compute_cmd_list2->Reset(compute_cmd_allocator2.Get(), nullptr);
            compute_cmd_list2->CopyResource(compute_readback_buffer.Get(), compute_buffer.Get());
            compute_cmd_list2->Close();
            ID3D12CommandList* copy_lists[] = { compute_cmd_list2.Get() };
            command_queue->ExecuteCommandLists(_countof(copy_lists), copy_lists);
            wait_for_gpu(command_queue, fence_obj, fence_event, fence_val);
            check_done = true;
        }
        swap_chain->Present(1, 0);
        frame_index = swap_chain->GetCurrentBackBufferIndex();
        wait_for_gpu(command_queue, fence_obj, fence_event, fence_val);
    }

    CloseHandle(fence_event);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
