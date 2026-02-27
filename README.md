# GPU Compute (Unity)
GPU Compute provides the ultimate & easiest way to setup & execute GPU compute shaders in Unity. Reduces complexity and boilerplate code while providing powerful features for GPU memory management, performance tracking, and shader execution control.

## Features

- **Simplified Shader Execution** - Easy setup and execution of compute shaders (synchronous & asynchronous)
- **Automatic Buffer Management** - Create, edit, and read buffers with automatically calculated strides and lengths
- **Flexible Thread Group Sizing** - Automatically calculate optimal GPU thread group sizes for 1D, 2D, and 3D workloads
- **GPU Memory Tracking** - Track local and global GPU memory usage in real-time with formatted output
- **Performance Monitoring** - Measure compute shader execution time
- **Async GPU Readback** - Efficiently retrieve data from GPU using AsyncGPUReadbackRequest
- **Render Texture Support** - Full support for creating and managing render textures
- **Global Resources** - Centralized management of global buffers and textures accessible across multiple compute shader instances
- **Buffer Resizing** - Dynamically resize buffers without recreating instances
- **Shader Keywords** - Enable/disable local shader keywords programmatically
- **Buffer Debugging** - Retrieve and inspect buffer contents for debugging purposes
- **Centralized in a Single Class** - All GPU compute operations managed through one easy-to-use class

![alt text](https://github.com/Aelstraz/Unity-GPU-Compute/blob/main/Screenshot.png?raw=true)

---

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Buffers & Textures](#buffers--textures)
4. [Shader Variables](#shader-variables)
5. [Thread Group Sizes](#thread-group-sizes)
6. [Execution](#execution)
7. [Memory & Performance](#memory--performance)
8. [Async Operations](#async-operations)
9. [Global Resources](#global-resources)
10. [Buffer Debugging](#buffer-debugging)
11. [Advanced Features](#advanced-features)

---

## Installation

Simply add the entire folder into your project's Assets folder.

---

## Quick Start

### Instantiate

Create a new instance of GPUCompute by providing your compute shader:

```csharp
GPUCompute gpuCompute = new GPUCompute(myComputeShader);
```

You can also specify a compute queue type for async execution:

```csharp
GPUCompute gpuCompute = new GPUCompute(myComputeShader, ComputeQueueType.Default);
```

### Dispose

Always dispose of your GPUCompute instances when finished to prevent GPU memory leaks:

```csharp
gpuCompute.Dispose();              // Disposes both local and global resources
gpuCompute.DisposeLocal();         // Disposes only local buffers/textures
GPUCompute.DisposeGlobal();        // Disposes only global buffers/textures
```

---

## Buffers & Textures

### Creating & Setting Buffers

Create buffers from existing data (stride and length are calculated automatically):

```csharp
Vector3[] vertices = new Vector3[1000];
gpuCompute.SetBuffer("vertices", ref vertices);

float[] uvs = new float[2000];
gpuCompute.SetBuffer("uvs", ref uvs);
```

Create empty buffers by specifying the struct type and length:

```csharp
gpuCompute.CreateEmptyBuffer<int>("myEmptyBuffer", 500);
gpuCompute.CreateEmptyBuffer<Vector3>("positions", 1000, kernelIndex: 0);
```

Set buffers with multiple kernel indices:

```csharp
int[] kernelIndices = { 0, 1, 2 };
gpuCompute.SetBuffer("sharedData", ref data, kernelIndices);
```

Specify buffer type and mode:

```csharp
gpuCompute.CreateEmptyBuffer<float>(
    "counterBuffer", 
    100, 
    ComputeBufferType.Counter, 
    ComputeBufferMode.Dynamic
);
```

### Reading Buffer Data

Retrieve buffer data from GPU to CPU:

```csharp
Vector3[] resultBuffer = new Vector3[100];
gpuCompute.GetBufferData("vertices", ref resultBuffer);

List<Vector3> resultList = new List<Vector3>(100);
gpuCompute.GetBufferData("vertices", ref resultList);

NativeArray<Vector3> resultNative = new NativeArray<Vector3>(100, Allocator.Persistent);
gpuCompute.GetBufferData("vertices", ref resultNative);
```

### Resizing Buffers

Dynamically resize buffers (data will be lost):

```csharp
gpuCompute.ResizeBuffer("myBuffer", 2000);
```

### Render Textures

Create render textures for compute shader output:

```csharp
RenderTextureDescriptor descriptor = new RenderTextureDescriptor(1024, 1024, RenderTextureFormat.ARGBFloat);
gpuCompute.CreateEmptyRenderTexture("outputTexture", descriptor);

// With mip levels and custom settings
gpuCompute.CreateEmptyRenderTexture(
    "outputTexture",
    descriptor,
    kernelIndex: 0,
    mipLevel: 0,
    wrapMode: TextureWrapMode.Clamp,
    filterMode: FilterMode.Point,
    anisoLevel: 1
);
```

Set render textures:

```csharp
RenderTexture myTexture = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGBFloat);
gpuCompute.SetRenderTexture("myRenderTexture", ref myTexture);
```

Retrieve render texture data:

```csharp
Texture2D outputTexture = new Texture2D(1024, 1024, TextureFormat.RGBAFloat, false);
gpuCompute.GetRenderTextureData("outputTexture", ref outputTexture);
```

### Texture Support

Set various texture types:

```csharp
Texture2D texture2D = Resources.Load<Texture2D>("myTexture");
gpuCompute.SetTexture("inputTexture", ref texture2D);

Texture3D texture3D = Resources.Load<Texture3D>("myVolume");
gpuCompute.SetTexture("volumeTexture", ref texture3D);

Texture2DArray textureArray = Resources.Load<Texture2DArray>("myTextureArray");
gpuCompute.SetTextureArray("textureArrayInput", ref textureArray);
```

---

## Shader Variables

### Scalar Values

```csharp
gpuCompute.SetInt("count", 100);
gpuCompute.SetFloat("deltaTime", Time.deltaTime);
gpuCompute.SetBool("useOptimization", true);
```

### Vectors

```csharp
gpuCompute.SetVector("position", new Vector3(1, 2, 3));
gpuCompute.SetVector("colorData", new Vector4(1, 0, 0, 1));
gpuCompute.SetVector("uv", new Vector2(0.5f, 0.5f));
gpuCompute.SetVector("gridSize", new Vector3Int(10, 10, 10));
```

### Arrays

```csharp
float[] floatArray = new float[100];
gpuCompute.SetFloatArray("floatData", floatArray);

Vector4[] vectorArray = new Vector4[50];
gpuCompute.SetVectorArray("vectorData", vectorArray);
```

### Matrices

```csharp
Matrix4x4 transformMatrix = Matrix4x4.identity;
gpuCompute.SetMatrix("transform", transformMatrix);

Matrix4x4[] matrixArray = new Matrix4x4[10];
gpuCompute.SetMatrixArray("transforms", matrixArray);
```

### Retrieving Variable Values

```csharp
int value = gpuCompute.GetInt("myInt");
float value = gpuCompute.GetFloat("myFloat");
Vector4 vector = gpuCompute.GetVector("myVector");
Matrix4x4 matrix = gpuCompute.GetMatrix("myMatrix");
```

---

## Thread Group Sizes

### Manual Thread Group Setting

```csharp
gpuCompute.SetThreadGroupSize(new Vector3Int(8, 8, 1));
Vector3Int currentSize = gpuCompute.GetThreadGroupSize();
```

### Automatic 1D Thread Group Calculation

For array-based workloads:

```csharp
int jobLength = vertices.Length;
int shaderThreads = 256;
gpuCompute.SetCalculatedThreadGroupSize(jobLength, kernelIndex: 0);
```

### Automatic 2D Thread Group Calculation

For texture-based workloads:

```csharp
int width = 1024;
int height = 1024;
gpuCompute.SetCalculatedThreadGroupSize(width, height, kernelIndex: 0);
```

### Automatic 3D Thread Group Calculation

For volume-based workloads:

```csharp
int width = 64;
int height = 64;
int depth = 64;
gpuCompute.SetCalculatedThreadGroupSize(width, height, depth, kernelIndex: 0);
```

### Getting Kernel Thread Group Sizes

Retrieve the thread group sizes defined in your compute shader kernel:

```csharp
Vector3Int kernelGroupSizes = gpuCompute.GetKernelThreadGroupSizes(kernelIndex: 0);
Debug.Log($"Kernel thread group size: {kernelGroupSizes}");
```

---

## Execution

### Synchronous Execution

Execute the compute shader and wait for completion:

```csharp
gpuCompute.Execute(kernelIndex: 0);
```

### Asynchronous Execution

Execute the compute shader asynchronously (DX12 only):

```csharp
StartCoroutine(gpuCompute.ExecuteAsync(kernelIndex: 0));
```

Subscribe to completion events:

```csharp
gpuCompute.OnExecuteComplete += OnComputeComplete;

private void OnComputeComplete(int kernelIndex)
{
    Debug.Log($"Kernel {kernelIndex} execution completed");
}
```

### Check Execution Status

```csharp
if (gpuCompute.IsExecuting())
{
    Debug.Log("Compute shader is currently executing");
}
```

---

## Memory & Performance

### Track GPU Memory Usage

Get local GPU memory used by the current instance:

```csharp
long localMemoryBytes = gpuCompute.GetLocalGPUMemoryUsed();
string localMemoryFormatted = gpuCompute.GetLocalGPUMemoryUsedFormatted();
// Output: "1.25 MB"
```

Get global GPU memory used across all instances:

```csharp
long globalMemoryBytes = GPUCompute.GetGlobalGPUMemoryUsed();
string globalMemoryFormatted = GPUCompute.GetGlobalGPUMemoryUsedFormatted();
```

### Track Execution Time

Measure the duration of the last compute shader execution:

```csharp
TimeSpan lastExecutionTime = gpuCompute.GetLastComputeTime();
Debug.Log($"Last execution took: {lastExecutionTime.TotalMilliseconds} ms");
```

### Format Byte Counts

Convert byte counts to human-readable strings:

```csharp
string formatted = GPUCompute.ByteCountToFormattedString(1024000);
// Output: "1.02 MB"
```

---

## Async Operations

### Async Buffer Readback

Asynchronously retrieve buffer data using AsyncGPUReadbackRequest for better performance:

```csharp
StartCoroutine(gpuCompute.GetBufferDataAsync("myBuffer"));

// Subscribe to readback completion
gpuCompute.OnReadbackComplete += OnBufferReadbackComplete;

private void OnBufferReadbackComplete(AsyncGPUReadbackRequest request, string bufferName)
{
    Vector3[] data = new Vector3[request.width / 12]; // 12 bytes per Vector3
    GPUCompute.ReadbackRequestToArray(ref request, ref data);
    // Process data...
}
```

Read buffer data with offset and length:

```csharp
StartCoroutine(gpuCompute.GetBufferDataAsync("myBuffer", length: 100, startIndex: 50));
```

### Async Render Texture Readback

Asynchronously retrieve render texture data:

```csharp
StartCoroutine(gpuCompute.GetRenderTextureDataAsync("outputTexture"));

gpuCompute.OnReadbackComplete += OnTextureReadbackComplete;

private void OnTextureReadbackComplete(AsyncGPUReadbackRequest request, string textureName)
{
    Texture2D output = new Texture2D(request.width, request.height, TextureFormat.RGBAFloat, false);
    GPUCompute.ReadbackRequestToTexture2D(ref request, ref output);
}
```

Retrieve specific regions:

```csharp
StartCoroutine(gpuCompute.GetRenderTextureDataAsync(
    "outputTexture",
    width: 256,
    height: 256,
    depth: 1,
    mipIndex: 0,
    x: 100,
    y: 100,
    z: 0
));
```

### Readback Data Conversion

Convert async readback requests to different formats:

```csharp
// To NativeArray
NativeArray<Vector3> nativeData = new NativeArray<Vector3>(100, Allocator.Persistent);
GPUCompute.ReadbackRequestToNativeArray(ref request, ref nativeData);

// To List
List<Vector3> listData = new List<Vector3>(100);
GPUCompute.ReadbackRequestToList(ref request, ref listData);

// To Texture3D
Texture3D volume = new Texture3D(64, 64, 64, TextureFormat.RGBAFloat, false);
GPUCompute.ReadbackRequestToTexture3D(ref request, ref volume);
```

---

## Global Resources

Global buffers and textures are accessible across all GPUCompute instances and persist for the lifetime of the application.

### Global Buffers

Create and set global buffers:

```csharp
Vector3[] globalVertices = new Vector3[1000];
GPUCompute.SetGlobalBuffer("globalVertices", ref globalVertices);

GPUCompute.CreateEmptyGlobalBuffer<float>("globalData", 5000);
```

Retrieve global buffer data:

```csharp
Vector3[] outputBuffer = new Vector3[1000];
GPUCompute.GetGlobalBufferData("globalVertices", ref outputBuffer);
```

Resize global buffers:

```csharp
GPUCompute.ResizeGlobalBuffer("globalData", 10000);
```

### Global Textures

Set global textures:

```csharp
Texture2D globalTexture = Resources.Load<Texture2D>("myGlobalTexture");
GPUCompute.SetGlobalTexture("globalTextureName", ref globalTexture);

Texture3D globalVolume = Resources.Load<Texture3D>("myVolume");
GPUCompute.SetGlobalTexture("globalVolume", ref globalVolume);

Texture2DArray globalArray = Resources.Load<Texture2DArray>("myArray");
GPUCompute.SetGlobalTextureArray("globalTextureArray", ref globalArray);
```

Create global render textures:

```csharp
RenderTextureDescriptor descriptor = new RenderTextureDescriptor(2048, 2048, RenderTextureFormat.ARGBFloat);
GPUCompute.CreateEmptyGlobalRenderTexture("globalOutput", descriptor);
```

Retrieve global texture data:

```csharp
Texture2D output = new Texture2D(2048, 2048, TextureFormat.RGBAFloat, false);
GPUCompute.GetGlobalRenderTextureData("globalOutput", ref output);
```

### Linking Global Resources

Link global resources to compute shader instances:

```csharp
gpuCompute.LinkGlobalBuffer("globalBufferName", "globalBufferName", kernelIndex: 0);
gpuCompute.LinkGlobalTexture("globalTextureName", "globalTextureName", kernelIndex: 0);
gpuCompute.LinkGlobalRenderTexture("globalOutputName", "globalOutputName", kernelIndex: 0);
```

Link to multiple kernels:

```csharp
int[] kernelIndices = { 0, 1, 2 };
gpuCompute.LinkGlobalBuffer("buffer", "buffer", kernelIndices);
```

### Dispose Global Resources

```csharp
GPUCompute.DisposeGlobalBuffer("globalBufferName");
GPUCompute.DisposeGlobalRenderTexture("globalTextureName");
GPUCompute.DisposeGlobal(); // Dispose all global resources
```

---

## Shader Keywords

### Enable/Disable Keywords

Enable shader keywords:

```csharp
gpuCompute.EnableKeyword("MY_KEYWORD");
```

Disable shader keywords:

```csharp
gpuCompute.DisableKeyword("MY_KEYWORD");
```

Set keyword state by boolean:

```csharp
LocalKeyword keyword = gpuCompute.GetKeywordSpace().FindKeyword("MY_KEYWORD");
gpuCompute.SetKeyword(keyword, true);
```

### Query Keyword State

Check if keyword is enabled:

```csharp
if (gpuCompute.IsKeywordEnabled("MY_KEYWORD"))
{
    Debug.Log("MY_KEYWORD is enabled");
}
```

Get all enabled keywords:

```csharp
LocalKeyword[] enabledKeywords = gpuCompute.GetEnabledKeywords();
foreach (var keyword in enabledKeywords)
{
    Debug.Log($"Enabled: {keyword.name}");
}
```

---

## Buffer Debugging

Debug buffers to inspect their contents (CPU-side only, for debugging):

```csharp
string debugInfo = gpuCompute.DebugBuffer<Vector3>("myBuffer");
Debug.Log(debugInfo);
// Output:
// Buffer Name: [myBuffer], Data Type: [System.Numerics.Vector3], 
// Length: [100], Stride: [12], VRAM Usage: [1.17 KB], ...
// Values:
// Index 0: (1.00, 2.00, 3.00)
// Index 1: (4.00, 5.00, 6.00)
// ...
```

Debug global buffers:

```csharp
string debugInfo = GPUCompute.DebugGlobalBuffer<float>("globalBuffer");
Debug.Log(debugInfo);
```

---

## Advanced Features

### Kernel Management

Find kernel indices:

```csharp
int kernelIndex = gpuCompute.FindKernel("CSMain");
```

Check if kernel exists:

```csharp
if (gpuCompute.HasKernel("CSMain"))
{
    Debug.Log("CSMain kernel found");
}
```

Check device support:

```csharp
if (gpuCompute.IsSupported(kernelIndex: 0))
{
    Debug.Log("Kernel is supported on this device");
}
```

### Compute Queue Configuration

Set compute queue type (for async execution):

```csharp
gpuCompute.SetComputeQueueType(ComputeQueueType.Default);
ComputeQueueType currentType = gpuCompute.GetComputeQueueType();
```

Set synchronization stage flags:

```csharp
gpuCompute.SetSynchronisationStageFlags(SynchronisationStageFlags.ComputeProcessing);
SynchronisationStageFlags flags = gpuCompute.GetSynchronisationStageFlags();
```

### Accessing Compute Shader

```csharp
ComputeShader shader = gpuCompute.GetComputeShader();
```

### Get Keyword Space

```csharp
LocalKeywordSpace keywordSpace = gpuCompute.GetKeywordSpace();
```

---

## Example Usage

Complete example demonstrating common workflow:

```csharp
using UnityEngine;
using GPUComputeModule;

public class ComputeExample : MonoBehaviour
{
    private GPUCompute gpuCompute;
    private ComputeShader computeShader;
    
    void Start()
    {
        // Initialize
        computeShader = Resources.Load<ComputeShader>("MyComputeShader");
        gpuCompute = new GPUCompute(computeShader);
        
        // Setup data
        Vector3[] vertices = new Vector3[1000];
        for (int i = 0; i < vertices.Length; i++)
            vertices[i] = new Vector3(i, 0, 0);
        
        // Create buffer and set data
        gpuCompute.SetBuffer("vertices", ref vertices);
        
        // Set shader variables
        gpuCompute.SetFloat("time", Time.deltaTime);
        gpuCompute.SetInt("vertexCount", vertices.Length);
        
        // Calculate thread groups for 1D workload
        gpuCompute.SetCalculatedThreadGroupSize(vertices.Length, kernelIndex: 0);
        
        // Execute synchronously
        gpuCompute.Execute();
        
        // Read results
        Vector3[] results = new Vector3[1000];
        gpuCompute.GetBufferData("vertices", ref results);
        
        // Check memory usage
        Debug.Log($"GPU Memory Used: {gpuCompute.GetLocalGPUMemoryUsedFormatted()}");
        Debug.Log($"Execution Time: {gpuCompute.GetLastComputeTime().TotalMilliseconds} ms");
        
        // Cleanup
        gpuCompute.Dispose();
    }
}
```

---

## Twitter: [@aelstraz](https://x.com/Aelstraz)