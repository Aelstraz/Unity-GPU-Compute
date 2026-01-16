using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace GPUComputeModule
{
    /*
     * GPUCompute provides an easy way to setup & execute GPU compute shaders in Unity. 
     * Create and manage buffers, track GPU memory usage & execution time, 
     * automatically calculate thread group sizes & buffer strides- all in one class.
     * Remember to call Dispose() when finished to prevent memory leaks on the GPU.
     * 
     * Author(s): Aelstraz
     */
    public class GPUCompute : IDisposable
    {
        private ComputeQueueType computeQueueType = ComputeQueueType.Background;
        private SynchronisationStageFlags synchronisationStageFlags = SynchronisationStageFlags.ComputeProcessing;
        private ComputeShader computeShader;
        private CommandBuffer commandBuffer;
        private GraphicsFence graphicsFence;
        private List<BufferInfo> buffers = new List<BufferInfo>();
        private static List<BufferInfo> globalBuffers = new List<BufferInfo>();
        private List<int> linkedGlobalBuffers = new List<int>();
        private bool isExecuting = false;
        private long localGPUMemoryUsage = 0;
        private static long globalGPUMemoryUsage = 0;
        private List<string> trackedMemoryObjectNames = new List<string>();
        private static List<string> trackedGlobalMemoryObjectNames = new List<string>();
        private TimeSpan lastComputeTime = TimeSpan.Zero;
        private Vector3Int threadGroupSize = new Vector3Int(1, 1, 1);

        public delegate void OnExecuteCompleteDelegate(int kernelIndex);
        /// <summary>
        /// Callback event for when the compute shader has finished executing (for async execution)
        /// </summary>
        public event OnExecuteCompleteDelegate OnExecuteComplete;

        /// <summary>
        /// Creates a new GPUCompute instance. Remember to call Dispose() before the application closes to avoid memory leaks on the GPU, or when finished to free up GPU memory
        /// </summary>
        /// <param name="computeShader">The compute shader that will be used</param>
        /// <param name="computeQueueType">Optional. Defines how the GPU will queue the execution of the compute shader</param>
        public GPUCompute(ComputeShader computeShader, ComputeQueueType computeQueueType = ComputeQueueType.Background)
        {
            this.computeShader = computeShader;
            this.computeQueueType = computeQueueType;

            commandBuffer = new CommandBuffer();
        }

        /// <summary>
        /// Executes the compute shader synchrously on the GPU. Ensure that the thread group size is set correctly before executing
        /// </summary>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <returns>A reference to the coroutine</returns>
        public void Execute(int kernelIndex = 0)
        {
            if (kernelIndex < 0)
            {
                Debug.LogError("Invalid kernel index: '" + kernelIndex + "', must be 0 or greater");
                return;
            }

            if (!isExecuting)
            {
                isExecuting = true;
                SetupCommandBuffer(kernelIndex, CommandBufferExecutionFlags.None);
                DateTime time = DateTime.Now;
                Graphics.ExecuteCommandBuffer(commandBuffer);
                ExecuteCompleted(kernelIndex, DateTime.Now - time);
            }
        }

        /// <summary>
        /// Executes the compute shader asynchrously on the GPU via StartCoroutine(). Ensure that the thread group size is set correctly before executing
        /// </summary>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <returns>A reference to the coroutine</returns>
        public IEnumerator ExecuteAsync(int kernelIndex = 0)
        {
            if (kernelIndex < 0)
            {
                Debug.LogError("Invalid kernel index: '" + kernelIndex + "', must be 0 or greater");
                yield return null;
            }

            if (!isExecuting)
            {
                isExecuting = true;
                SetupCommandBuffer(kernelIndex, CommandBufferExecutionFlags.AsyncCompute);
                DateTime time = DateTime.Now;
                Graphics.ExecuteCommandBufferAsync(commandBuffer, computeQueueType);
                yield return new WaitWhile(() => !graphicsFence.passed);
                ExecuteCompleted(kernelIndex, DateTime.Now - time);
            }
        }

        /// <summary>
        /// Manually sets the thread group size for the compute shader
        /// </summary>
        /// <param name="threadGroupSize">The thread group size to set</param>
        public void SetThreadGroupSize(Vector3Int threadGroupSize)
        {
            threadGroupSize.x = Mathf.Max(1, threadGroupSize.x);
            threadGroupSize.y = Mathf.Max(1, threadGroupSize.y);
            threadGroupSize.z = Mathf.Max(1, threadGroupSize.z);
            this.threadGroupSize = threadGroupSize;
        }

        /// <summary>
        /// Returns the thread group size for the compute shader
        /// </summary>
        /// <returns>The thread group size</returns>
        public Vector3Int GetThreadGroupSize()
        {
            return threadGroupSize;
        }

        /// <summary>
        /// Automatically calculates and sets the thread group size for a one dimensional workload (e.g. an array)
        /// </summary>
        /// <param name="jobLength">The total number of items to be processed in your workload (e.g. length of an array)</param>
        /// <param name="shaderNumberOfThreads">The number of threads as declared in your compute shader</param>
        /// <returns>The total width of the threads and group sizes (for use in your compute shader to get the index of the dispatch thread)</returns>
        public int SetCalculatedThreadGroupSize(int jobLength, Vector3Int shaderNumberOfThreads)
        {
            jobLength = Mathf.Max(1, jobLength);
            shaderNumberOfThreads.x = Mathf.Max(1, shaderNumberOfThreads.x);
            shaderNumberOfThreads.y = Mathf.Max(1, shaderNumberOfThreads.y);
            shaderNumberOfThreads.z = Mathf.Max(1, shaderNumberOfThreads.z);

            int totalNumberOfShaderThreads = shaderNumberOfThreads.x * shaderNumberOfThreads.y * shaderNumberOfThreads.z;
            int newGroupSize = Mathf.Max(1, Mathf.CeilToInt((float)jobLength / (float)totalNumberOfShaderThreads));
            threadGroupSize = new Vector3Int(newGroupSize, 1, 1);
            return threadGroupSize.x * shaderNumberOfThreads.x;
        }

        /// <summary>
        /// Automatically calculates and sets the thread group size for a two dimensional workload (e.g. a texture)
        /// </summary>
        /// <param name="width">The width of the workload</param>
        /// <param name="height">The height of the workload</param>
        /// <param name="shaderNumberOfThreads">The number of threads as declared in your compute shader</param>
        public void SetCalculatedThreadGroupSize2D(int width, int height, Vector3Int shaderNumberOfThreads)
        {
            width = Mathf.Max(1, width);
            height = Mathf.Max(1, height);
            shaderNumberOfThreads.x = Mathf.Max(1, shaderNumberOfThreads.x);
            shaderNumberOfThreads.y = Mathf.Max(1, shaderNumberOfThreads.y);
            shaderNumberOfThreads.z = Mathf.Max(1, shaderNumberOfThreads.z);

            threadGroupSize.x = Mathf.CeilToInt((float)width / shaderNumberOfThreads.x);
            threadGroupSize.y = Mathf.CeilToInt((float)height / shaderNumberOfThreads.y);
            threadGroupSize.z = 1;
        }

        /// <summary>
        /// Automatically calculates and sets the thread group size for a three dimensional workload (e.g. a volume)
        /// </summary>
        /// <param name="width">The width of the workload</param>
        /// <param name="height">The height of the workload</param>
        /// <param name="depth">The depth of the workload</param>
        /// <param name="shaderNumberOfThreads">The number of threads as declared in your compute shader</param>
        public void SetCalculatedThreadGroupSize3D(int width, int height, int depth, Vector3Int shaderNumberOfThreads)
        {
            width = Mathf.Max(1, width);
            height = Mathf.Max(1, height);
            depth = Mathf.Max(1, depth);
            shaderNumberOfThreads.x = Mathf.Max(1, shaderNumberOfThreads.x);
            shaderNumberOfThreads.y = Mathf.Max(1, shaderNumberOfThreads.y);
            shaderNumberOfThreads.z = Mathf.Max(1, shaderNumberOfThreads.z);

            threadGroupSize.x = Mathf.CeilToInt((float)width / shaderNumberOfThreads.x);
            threadGroupSize.y = Mathf.CeilToInt((float)height / shaderNumberOfThreads.y);
            threadGroupSize.z = Mathf.CeilToInt((float)depth / shaderNumberOfThreads.z);
        }

        private void SetupCommandBuffer(int kernelIndex, CommandBufferExecutionFlags commandBufferExecutionFlags)
        {
            commandBuffer.Clear();
            commandBuffer.SetExecutionFlags(commandBufferExecutionFlags);
            commandBuffer.DispatchCompute(computeShader, kernelIndex, threadGroupSize.x, threadGroupSize.y, threadGroupSize.z);
            graphicsFence = commandBuffer.CreateGraphicsFence(GraphicsFenceType.CPUSynchronisation, synchronisationStageFlags);
        }

        private void ExecuteCompleted(int kernelIndex, TimeSpan computeTime)
        {
            isExecuting = false;
            lastComputeTime = computeTime;

            if (OnExecuteComplete != null)
            {
                OnExecuteComplete(kernelIndex);
            }
        }

        /// <summary>
        /// Sets the value of an int variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetInt(string name, int value)
        {
            AddStructMemoryUsage<int>(name);
            computeShader.SetInt(name, value);
        }

        /// <summary>
        /// Sets the value of a float variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetFloat(string name, float value)
        {
            AddStructMemoryUsage<float>(name);
            computeShader.SetFloat(name, value);
        }

        /// <summary>
        /// Sets the value of a bool variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetBool(string name, bool value)
        {
            AddStructMemoryUsage<bool>(name);
            computeShader.SetBool(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector4 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector4 value)
        {
            AddStructMemoryUsage<Vector4>(name);
            computeShader.SetVector(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector3 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector3 value)
        {
            AddStructMemoryUsage<Vector3>(name);
            computeShader.SetVector(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector2 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector2 value)
        {
            AddStructMemoryUsage<Vector2>(name);
            computeShader.SetVector(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector3Int variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector3Int value)
        {
            AddStructMemoryUsage<Vector3Int>(name);
            computeShader.SetVector(name, new Vector3(value.x, value.y, value.z));
        }

        /// <summary>
        /// Sets the value of a Vector2Int variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector2Int value)
        {
            AddStructMemoryUsage<Vector2Int>(name);
            computeShader.SetVector(name, new Vector2(value.x, value.y));
        }

        /// <summary>
        /// Sets the value of a Matrix4x4 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetMatrix(string name, Matrix4x4 value)
        {
            AddStructMemoryUsage<Matrix4x4>(name);
            computeShader.SetMatrix(name, value);
        }

        /// <summary>
        /// Sets the value of a Texture variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTexture(string name, Texture value, int kernelIndex = 0, int mipLevel = 0)
        {
            SetTexture(name, value, new int[] { kernelIndex }, mipLevel);
        }

        /// <summary>
        /// Sets the value of a Texture variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the texture in the compute shader</param>
        /// <param name="value">The value to set</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTexture(string name, Texture value, int[] kernelIndices, int mipLevel = 0)
        {
            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTexture(kernelIndex, name, value, mipLevel);
            }
            AddTextureMemoryUsage(name, ref value);
        }

        /// <summary>
        /// Sets the value of a Texture variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void SetRenderTexture(string name, RenderTexture value, int kernelIndex = 0, int mipLevel = 0, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            SetRenderTexture(name, value, new int[] { kernelIndex }, mipLevel, renderTextureSubElement);
        }

        /// <summary>
        /// Sets the value of a Texture variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the texture in the compute shader</param>
        /// <param name="value">The value to set</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void SetRenderTexture(string name, RenderTexture value, int[] kernelIndices, int mipLevel = 0, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTexture(kernelIndex, name, value, mipLevel, renderTextureSubElement);
            }
            AddGlobalTextureMemoryUsage(name, ref value);
        }

        /// <summary>
        /// Links an existing global texture to this compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="globalTextureName">The global name of the texture</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        public void LinkGlobalTexture(string name, string globalTextureName, int kernelIndex = 0)
        {
            LinkGlobalTexture(name, globalTextureName, new int[] { kernelIndex });
        }

        /// <summary>
        /// Links an existing global texture to this compute shader
        /// </summary>
        /// <param name="name">The name of the keyword in the compute shader</param>
        /// <param name="globalTextureName">The global name of the texture</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        public void LinkGlobalTexture(string name, string globalTextureName, int[] kernelIndices)
        {
            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTextureFromGlobal(kernelIndex, name, globalTextureName);
            }
            Texture globalTexture = Shader.GetGlobalTexture(globalTextureName);
            AddGlobalTextureMemoryUsage(name, ref globalTexture);
        }

        /// <summary>
        /// Sets a keyword value in the compute shader
        /// </summary>
        /// <param name="keyword">The keyword to modify</param>
        /// <param name="value">The value to set</param>
        public void SetKeyword(LocalKeyword keyword, bool value)
        {
            computeShader.SetKeyword(keyword, value);
        }

        /// <summary>
        /// Sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        public void SetBufferData<T>(string name, ref NativeArray<T> data) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref buffers);
            if (CheckIfBufferInfoIsValid<T>(name, data.Length, bufferData))
            {
                bufferData.buffer.SetData(data);
            }
        }

        /// <summary>
        /// Sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        public void SetBufferData<T>(string name, ref List<T> data) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref buffers);
            if (CheckIfBufferInfoIsValid<T>(name, data.Count, bufferData))
            {
                bufferData.buffer.SetData(data);
            }
        }

        /// <summary>
        /// Sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        public void SetBufferData<T>(string name, ref T[] data) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref buffers);
            if (CheckIfBufferInfoIsValid<T>(name, data.Length, bufferData))
            {
                bufferData.buffer.SetData(data);
            }
        }

        /// <summary>
        /// Sets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        public static void SetGlobalBufferData<T>(string name, ref NativeArray<T> data) where T : struct
        {
            BufferInfo bufferInfo = GetBufferFromList(name, ref globalBuffers);
            if (CheckIfBufferInfoIsValid<T>(name, data.Length, bufferInfo))
            {
                bufferInfo.buffer.SetData(data);
            }
        }

        /// <summary>
        /// Sets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        public static void SetGlobalBufferData<T>(string name, ref List<T> data) where T : struct
        {
            BufferInfo bufferInfo = GetBufferFromList(name, ref globalBuffers);
            if (CheckIfBufferInfoIsValid<T>(name, data.Count, bufferInfo))
            {
                bufferInfo.buffer.SetData(data);
            }
        }

        /// <summary>
        /// Sets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        public static void SetGlobalBufferData<T>(string name, ref T[] data) where T : struct
        {
            BufferInfo bufferInfo = GetBufferFromList(name, ref globalBuffers);
            if (CheckIfBufferInfoIsValid<T>(name, data.Length, bufferInfo))
            {
                bufferInfo.buffer.SetData(data);
            }
        }

        /// <summary>
        /// Creates and sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void SetBuffer<T>(string name, ref T[] data, int kernelIndex = 0, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyBuffer<T>(name, data.Length, new int[] { kernelIndex }, computeBufferType, computeBufferMode);
            SetBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates and sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="kernelIndices">An array containing which kernel(s) to set the buffer to in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void SetBuffer<T>(string name, ref T[] data, int[] kernelIndices, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyBuffer<T>(name, data.Length, kernelIndices, computeBufferType, computeBufferMode);
            SetBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates and sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void SetBuffer<T>(string name, ref List<T> data, int kernelIndex = 0, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyBuffer<T>(name, data.Count, new int[] { kernelIndex }, computeBufferType, computeBufferMode);
            SetBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates and sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="kernelIndices">An array containing which kernel(s) to set the buffer to in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void SetBuffer<T>(string name, ref List<T> data, int[] kernelIndices, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyBuffer<T>(name, data.Count, kernelIndices, computeBufferType, computeBufferMode);
            SetBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates and sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void SetBuffer<T>(string name, ref NativeArray<T> data, int kernelIndex = 0, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyBuffer<T>(name, data.Length, new int[] { kernelIndex }, computeBufferType, computeBufferMode);
            SetBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates and sets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="kernelIndices">An array containing which kernel(s) to set the buffer to in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void SetBuffer<T>(string name, ref NativeArray<T> data, int[] kernelIndices, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyBuffer<T>(name, data.Length, kernelIndices, computeBufferType, computeBufferMode);
            SetBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates an empty buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="bufferLength">The length of the buffer</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void CreateEmptyBuffer<T>(string name, int bufferLength, int kernelIndex = 0, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyBuffer<T>(name, bufferLength, new int[] { kernelIndex }, computeBufferType, computeBufferMode);
        }

        /// <summary>
        /// Creates an empty buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="bufferLength">The length of the buffer</param>
        /// <param name="kernelIndices">An array containing which kernel(s) to set the buffer to in the compute shader</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public void CreateEmptyBuffer<T>(string name, int bufferLength, int[] kernelIndices, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            if (ListContainsBuffer(name, ref buffers))
            {
                Debug.LogWarning("Buffer " + name + " has already been created");
                return;
            }

            int stride = GetStructSizeInBytes<T>();
            if (stride > 0)
            {
                ComputeBuffer computeBuffer = new ComputeBuffer(bufferLength, stride, computeBufferType, computeBufferMode);

                buffers.Add(new BufferInfo(name, computeBuffer, typeof(T)));
                foreach (int kernelIndex in kernelIndices)
                {
                    computeShader.SetBuffer(kernelIndex, name, computeBuffer);
                }
                AddBufferMemoryUsage(ref localGPUMemoryUsage, stride, bufferLength);
            }
        }

        /// <summary>
        /// Creates and sets the value of a global buffer
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public static void SetGlobalBuffer<T>(string name, ref T[] data, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyGlobalBuffer<T>(name, data.Length, computeBufferType, computeBufferMode);
            SetGlobalBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates and sets the value of a global buffer
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public static void SetGlobalBuffer<T>(string name, ref List<T> data, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyGlobalBuffer<T>(name, data.Count, computeBufferType, computeBufferMode);
            SetGlobalBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates and sets the value of a global buffer
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public static void SetGlobalBuffer<T>(string name, ref NativeArray<T> data, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            CreateEmptyGlobalBuffer<T>(name, data.Length, computeBufferType, computeBufferMode);
            SetGlobalBufferData<T>(name, ref data);
        }

        /// <summary>
        /// Creates en empty global buffer
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="data">The data to set the buffer to</param>
        /// <param name="computeBufferType">Optional. Sets the buffer type</param>
        /// <param name="computeBufferMode">Optional. Sets the buffer mode</param>
        public static void CreateEmptyGlobalBuffer<T>(string name, int bufferLength, ComputeBufferType computeBufferType = ComputeBufferType.Default, ComputeBufferMode computeBufferMode = ComputeBufferMode.Immutable) where T : struct
        {
            if (ListContainsBuffer(name, ref globalBuffers))
            {
                Debug.LogWarning("Global buffer " + name + " has already been created");
                return;
            }

            int stride = GetStructSizeInBytes<T>();
            if (stride > 0)
            {
                ComputeBuffer computeBuffer = new ComputeBuffer(bufferLength, stride, computeBufferType, computeBufferMode);
                globalBuffers.Add(new BufferInfo(name, computeBuffer, typeof(T)));
                AddBufferMemoryUsage(ref globalGPUMemoryUsage, stride, bufferLength);
            }
        }

        /// <summary>
        /// Links an existing global buffer to this compute shader
        /// </summary>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        public void LinkGlobalBuffer(string name, int kernelIndex = 0)
        {
            LinkGlobalBuffer(name, new int[] { kernelIndex });
        }

        /// <summary>
        /// Links an existing global buffer to this compute shader
        /// </summary>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="kernelIndices">An array containing which kernel(s) to set the global buffer to in the compute shader</param>
        public void LinkGlobalBuffer(string name, int[] kernelIndices)
        {
            int globalBufferIndex = IndexOfBufferInList(name, ref globalBuffers);
            if (globalBufferIndex == -1)
            {
                Debug.LogError("Unable to link global buffer " + name + ", buffer not found");
                return;
            }
            else if (linkedGlobalBuffers.Contains(globalBufferIndex))
            {
                Debug.LogWarning("Unable to link global buffer " + name + ", buffer has already been linked");
                return;
            }

            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetBuffer(kernelIndex, name, globalBuffers[globalBufferIndex].buffer);
            }
            linkedGlobalBuffers.Add(globalBufferIndex);
        }

        /// <summary>
        /// Gets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        public void GetBufferData<T>(string name, ref T[] output) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref buffers);
            if (string.IsNullOrEmpty(bufferData.name))
            {
                Debug.LogError("Unable to get buffer data " + name + ", buffer not found");
                return;
            }
            else if (bufferData.dataType != typeof(T))
            {
                Debug.LogError("Unable to get buffer data " + name + ", supplied type is incorrect. BufferType=" + bufferData.dataType + ", SuppliedType=" + typeof(T));
                return;
            }

            bufferData.buffer.GetData(output);
        }

        /// <summary>
        /// Gets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        public void GetBufferData<T>(string name, ref List<T> output) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref buffers);
            if (string.IsNullOrEmpty(bufferData.name))
            {
                Debug.LogError("Unable to get buffer data " + name + ", buffer not found");
                return;
            }
            else if (bufferData.dataType != typeof(T))
            {
                Debug.LogError("Unable to get buffer data " + name + ", supplied type is incorrect. BufferType=" + bufferData.dataType + ", SuppliedType=" + typeof(T));
                return;
            }

            T[] temp = new T[output.Count];
            bufferData.buffer.GetData(temp);
            for (int i = 0; i < temp.Length; i++)
            {
                output[i] = temp[i];
            }
        }

        /// <summary>
        /// Gets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        public void GetBufferData<T>(string name, ref NativeArray<T> output) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref buffers);
            if (string.IsNullOrEmpty(bufferData.name))
            {
                Debug.LogError("Unable to get buffer data " + name + ", buffer not found");
                return;
            }
            else if (bufferData.dataType != typeof(T))
            {
                Debug.LogError("Unable to get buffer data " + name + ", supplied type is incorrect. BufferType=" + bufferData.dataType + ", SuppliedType=" + typeof(T));
                return;
            }

            T[] temp = new T[output.Length];
            bufferData.buffer.GetData(temp);
            for (int i = 0; i < temp.Length; i++)
            {
                output[i] = temp[i];
            }
        }

        /// <summary>
        /// Gets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        public static void GetGlobalBufferData<T>(string name, ref T[] output) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref globalBuffers);
            if (string.IsNullOrEmpty(bufferData.name))
            {
                Debug.LogError("Unable to get global buffer data " + name + ", global buffer not found");
                return;
            }
            else if (bufferData.dataType != typeof(T))
            {
                Debug.LogError("Unable to get global buffer data " + name + ", supplied type is incorrect. BufferType=" + bufferData.dataType + ", SuppliedType=" + typeof(T));
                return;
            }

            bufferData.buffer.GetData(output);
        }

        /// <summary>
        /// Gets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        public static void GetGlobalBufferData<T>(string name, ref List<T> output) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref globalBuffers);
            if (string.IsNullOrEmpty(bufferData.name))
            {
                Debug.LogError("Unable to get global buffer data " + name + ", global buffer not found");
                return;
            }
            else if (bufferData.dataType != typeof(T))
            {
                Debug.LogError("Unable to get global buffer data " + name + ", supplied type is incorrect. BufferType=" + bufferData.dataType + ", SuppliedType=" + typeof(T));
                return;
            }

            T[] temp = new T[output.Count];
            bufferData.buffer.GetData(temp);
            for (int i = 0; i < temp.Length; i++)
            {
                output[i] = temp[i];
            }
        }

        /// <summary>
        /// Gets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        public static void GetGlobalBufferData<T>(string name, ref NativeArray<T> output) where T : struct
        {
            BufferInfo bufferData = GetBufferFromList(name, ref globalBuffers);
            if (string.IsNullOrEmpty(bufferData.name))
            {
                Debug.LogError("Unable to get global buffer data " + name + ", global buffer not found");
                return;
            }
            else if (bufferData.dataType != typeof(T))
            {
                Debug.LogError("Unable to get global buffer data " + name + ", supplied type is incorrect. BufferType=" + bufferData.dataType + ", SuppliedType=" + typeof(T));
                return;
            }

            T[] temp = new T[output.Length];
            bufferData.buffer.GetData(temp);
            for (int i = 0; i < temp.Length; i++)
            {
                output[i] = temp[i];
            }
        }

        private static bool CheckIfBufferInfoIsValid<T>(string name, int length, BufferInfo bufferInfo)
        {
            if (string.IsNullOrEmpty(bufferInfo.name))
            {
                Debug.LogError("Unable to set buffer " + name + ", buffer not found");
                return false;
            }
            else if (bufferInfo.dataType != typeof(T))
            {
                Debug.LogError("Unable to set buffer " + name + ", supplied type is incorrect. BufferType=" + bufferInfo.dataType + ", SuppliedType=" + typeof(T));
                return false;
            }
            else if (length != bufferInfo.buffer.count)
            {
                Debug.LogError("Unable to set buffer " + name + ", supplied data length is incorrect. BufferLength=" + bufferInfo.buffer.count + ", SuppliedLength=" + length);
                return false;
            }
            return true;
        }

        private static bool ListContainsBuffer(string name, ref List<BufferInfo> bufferDataList)
        {
            bool contains = false;
            foreach (BufferInfo buffer in bufferDataList)
            {
                if (buffer.name == name)
                {
                    contains = true;
                    break;
                }
            }
            return contains;
        }

        private static BufferInfo GetBufferFromList(string name, ref List<BufferInfo> bufferDataList)
        {
            BufferInfo bufferData = new BufferInfo(null, null, null);
            foreach (BufferInfo buffer in bufferDataList)
            {
                if (buffer.name == name)
                {
                    bufferData = buffer;
                    break;
                }
            }
            return bufferData;
        }

        private static int IndexOfBufferInList(string name, ref List<BufferInfo> bufferDataList)
        {
            int index = -1;
            for (int i = 0; i < bufferDataList.Count; i++)
            {
                if (bufferDataList[i].name == name)
                {
                    index = i;
                    break;
                }
            }
            return index;
        }

        private void AddStructMemoryUsage<T>(string name) where T : struct
        {
            if (!trackedMemoryObjectNames.Contains(name))
            {
                trackedMemoryObjectNames.Add(name);
                localGPUMemoryUsage += GetStructSizeInBytes<T>();
            }
        }

        private static void AddBufferMemoryUsage(ref long totalMemoryUsage, int stride, int length)
        {
            totalMemoryUsage += stride * length;
        }

        private void AddTextureMemoryUsage(string name, ref Texture texture)
        {
            if (!trackedMemoryObjectNames.Contains(name))
            {
                trackedMemoryObjectNames.Add(name);
                localGPUMemoryUsage += (texture.width * texture.height * GetGraphicsFormatBitsPerPixel(texture.graphicsFormat)) / 8;
            }
        }

        private static void AddGlobalTextureMemoryUsage(string name, ref RenderTexture texture)
        {
            if (!trackedGlobalMemoryObjectNames.Contains(name))
            {
                trackedGlobalMemoryObjectNames.Add(name);
                globalGPUMemoryUsage += (texture.width * texture.height * GetGraphicsFormatBitsPerPixel(texture.graphicsFormat)) / 8;
            }
        }

        private static void AddGlobalTextureMemoryUsage(string name, ref Texture texture)
        {
            if (!trackedGlobalMemoryObjectNames.Contains(name))
            {
                trackedGlobalMemoryObjectNames.Add(name);
                globalGPUMemoryUsage += (texture.width * texture.height * GetGraphicsFormatBitsPerPixel(texture.graphicsFormat)) / 8;
            }
        }

        private static int GetGraphicsFormatBitsPerPixel(GraphicsFormat format)
        {
#pragma warning disable CS0618 // Type or member is obsolete
            switch (format)
            {
                case GraphicsFormat.R8_SInt:
                case GraphicsFormat.R8_SNorm:
                case GraphicsFormat.R8_SRGB:
                case GraphicsFormat.R8_UInt:
                case GraphicsFormat.R8_UNorm:
                case GraphicsFormat.S8_UInt:
                case GraphicsFormat.YUV2:
                    return 8;
                case GraphicsFormat.D16_UNorm:
                case GraphicsFormat.R16_SFloat:
                case GraphicsFormat.R16_SInt:
                case GraphicsFormat.R16_SNorm:
                case GraphicsFormat.R16_UInt:
                case GraphicsFormat.R16_UNorm:
                case GraphicsFormat.R8G8_SInt:
                case GraphicsFormat.R8G8_SNorm:
                case GraphicsFormat.R8G8_SRGB:
                case GraphicsFormat.R8G8_UInt:
                case GraphicsFormat.R8G8_UNorm:
                case GraphicsFormat.A1R5G5B5_UNormPack16:
                case GraphicsFormat.B4G4R4A4_UNormPack16:
                case GraphicsFormat.B5G5R5A1_UNormPack16:
                case GraphicsFormat.B5G6R5_UNormPack16:
                case GraphicsFormat.R4G4B4A4_UNormPack16:
                case GraphicsFormat.R5G5B5A1_UNormPack16:
                case GraphicsFormat.R5G6B5_UNormPack16:
                    return 16;
                case GraphicsFormat.B8G8R8_SInt:
                case GraphicsFormat.B8G8R8_SNorm:
                case GraphicsFormat.B8G8R8_SRGB:
                case GraphicsFormat.B8G8R8_UInt:
                case GraphicsFormat.B8G8R8_UNorm:
                case GraphicsFormat.R8G8B8_SInt:
                case GraphicsFormat.R8G8B8_SNorm:
                case GraphicsFormat.R8G8B8_SRGB:
                case GraphicsFormat.R8G8B8_UInt:
                case GraphicsFormat.R8G8B8_UNorm:
                    return 24;
                case GraphicsFormat.A2B10G10R10_SIntPack32:
                case GraphicsFormat.A2B10G10R10_UIntPack32:
                case GraphicsFormat.A2B10G10R10_UNormPack32:
                case GraphicsFormat.A2R10G10B10_SIntPack32:
                case GraphicsFormat.A2R10G10B10_UIntPack32:
                case GraphicsFormat.A2R10G10B10_UNormPack32:
                case GraphicsFormat.A2R10G10B10_XRSRGBPack32:
                case GraphicsFormat.A2R10G10B10_XRUNormPack32:
                case GraphicsFormat.B10G11R11_UFloatPack32:
                case GraphicsFormat.B8G8R8A8_SInt:
                case GraphicsFormat.B8G8R8A8_SNorm:
                case GraphicsFormat.B8G8R8A8_SRGB:
                case GraphicsFormat.B8G8R8A8_UInt:
                case GraphicsFormat.B8G8R8A8_UNorm:
                case GraphicsFormat.D24_UNorm:
                case GraphicsFormat.D24_UNorm_S8_UInt:
                case GraphicsFormat.D32_SFloat:
                case GraphicsFormat.E5B9G9R9_UFloatPack32:
                case GraphicsFormat.R10G10B10_XRSRGBPack32:
                case GraphicsFormat.R10G10B10_XRUNormPack32:
                case GraphicsFormat.R16G16_SFloat:
                case GraphicsFormat.R16G16_SInt:
                case GraphicsFormat.R16G16_SNorm:
                case GraphicsFormat.R16G16_UInt:
                case GraphicsFormat.R16G16_UNorm:
                case GraphicsFormat.R32_SFloat:
                case GraphicsFormat.R32_SInt:
                case GraphicsFormat.R32_UInt:
                case GraphicsFormat.R8G8B8A8_SInt:
                case GraphicsFormat.R8G8B8A8_SNorm:
                case GraphicsFormat.R8G8B8A8_SRGB:
                case GraphicsFormat.R8G8B8A8_UInt:
                case GraphicsFormat.R8G8B8A8_UNorm:
                    return 32;
                case GraphicsFormat.R16G16B16_SFloat:
                case GraphicsFormat.R16G16B16_SInt:
                case GraphicsFormat.R16G16B16_SNorm:
                case GraphicsFormat.R16G16B16_UInt:
                case GraphicsFormat.R16G16B16_UNorm:
                    return 48;
                case GraphicsFormat.A10R10G10B10_XRSRGBPack32:
                case GraphicsFormat.A10R10G10B10_XRUNormPack32:
                case GraphicsFormat.D32_SFloat_S8_UInt:
                case GraphicsFormat.R16G16B16A16_SFloat:
                case GraphicsFormat.R16G16B16A16_SInt:
                case GraphicsFormat.R16G16B16A16_SNorm:
                case GraphicsFormat.R16G16B16A16_UInt:
                case GraphicsFormat.R16G16B16A16_UNorm:
                case GraphicsFormat.R32G32_SFloat:
                case GraphicsFormat.R32G32_SInt:
                case GraphicsFormat.R32G32_UInt:
                case GraphicsFormat.RGBA_DXT1_SRGB:
                case GraphicsFormat.RGBA_DXT1_UNorm:
                case GraphicsFormat.RGBA_PVRTC_2Bpp_SRGB:
                case GraphicsFormat.RGBA_PVRTC_2Bpp_UNorm:
                case GraphicsFormat.RGBA_PVRTC_4Bpp_SRGB:
                case GraphicsFormat.RGBA_PVRTC_4Bpp_UNorm:
                case GraphicsFormat.RGB_A1_ETC2_SRGB:
                case GraphicsFormat.RGB_A1_ETC2_UNorm:
                case GraphicsFormat.RGB_ETC2_SRGB:
                case GraphicsFormat.RGB_ETC2_UNorm:
                case GraphicsFormat.RGB_ETC_UNorm:
                case GraphicsFormat.RGB_PVRTC_2Bpp_SRGB:
                case GraphicsFormat.RGB_PVRTC_2Bpp_UNorm:
                case GraphicsFormat.RGB_PVRTC_4Bpp_SRGB:
                case GraphicsFormat.RGB_PVRTC_4Bpp_UNorm:
                case GraphicsFormat.R_BC4_SNorm:
                case GraphicsFormat.R_BC4_UNorm:
                case GraphicsFormat.R_EAC_SNorm:
                case GraphicsFormat.R_EAC_UNorm:
                    return 64;
                case GraphicsFormat.R32G32B32_SFloat:
                case GraphicsFormat.R32G32B32_SInt:
                case GraphicsFormat.R32G32B32_UInt:
                    return 96;
                case GraphicsFormat.R32G32B32A32_SFloat:
                case GraphicsFormat.R32G32B32A32_SInt:
                case GraphicsFormat.R32G32B32A32_UInt:
                case GraphicsFormat.RGBA_ASTC10X10_SRGB:
                case GraphicsFormat.RGBA_ASTC10X10_UFloat:
                case GraphicsFormat.RGBA_ASTC10X10_UNorm:
                case GraphicsFormat.RGBA_ASTC12X12_SRGB:
                case GraphicsFormat.RGBA_ASTC12X12_UFloat:
                case GraphicsFormat.RGBA_ASTC12X12_UNorm:
                case GraphicsFormat.RGBA_ASTC4X4_SRGB:
                case GraphicsFormat.RGBA_ASTC4X4_UFloat:
                case GraphicsFormat.RGBA_ASTC4X4_UNorm:
                case GraphicsFormat.RGBA_ASTC5X5_SRGB:
                case GraphicsFormat.RGBA_ASTC5X5_UFloat:
                case GraphicsFormat.RGBA_ASTC5X5_UNorm:
                case GraphicsFormat.RGBA_ASTC6X6_SRGB:
                case GraphicsFormat.RGBA_ASTC6X6_UFloat:
                case GraphicsFormat.RGBA_ASTC6X6_UNorm:
                case GraphicsFormat.RGBA_ASTC8X8_SRGB:
                case GraphicsFormat.RGBA_ASTC8X8_UFloat:
                case GraphicsFormat.RGBA_ASTC8X8_UNorm:
                case GraphicsFormat.RGBA_BC7_SRGB:
                case GraphicsFormat.RGBA_BC7_UNorm:
                case GraphicsFormat.RGBA_DXT3_SRGB:
                case GraphicsFormat.RGBA_DXT3_UNorm:
                case GraphicsFormat.RGBA_DXT5_SRGB:
                case GraphicsFormat.RGBA_DXT5_UNorm:
                case GraphicsFormat.RGBA_ETC2_SRGB:
                case GraphicsFormat.RGBA_ETC2_UNorm:
                case GraphicsFormat.RGB_BC6H_SFloat:
                case GraphicsFormat.RGB_BC6H_UFloat:
                case GraphicsFormat.RG_BC5_SNorm:
                case GraphicsFormat.RG_BC5_UNorm:
                case GraphicsFormat.RG_EAC_SNorm:
                case GraphicsFormat.RG_EAC_UNorm:
                    return 128;
            }
#pragma warning restore CS0618 // Type or member is obsolete
            return 0;
        }

        private static int GetStructSizeInBytes<T>() where T : struct
        {
            int stride = -1;
            try
            {
                stride = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            }
            catch (Exception e)
            {
                Debug.LogError("Unable to get stride value of struct: " + typeof(T).Name + ". " + e.Message);
            }
            return stride;
        }

        /// <summary>
        /// Returns the combined local and global GPU memory used as bytes
        /// </summary>
        /// <returns>The combined local and global GPU memory used as bytes</returns>
        public long GetTotalGPUMemoryUsed()
        {
            return localGPUMemoryUsage + globalGPUMemoryUsage;
        }

        /// <summary>
        /// Returns the combined local and global GPU memory used, calculated and formatted into a string showing the unit
        /// </summary>
        /// <returns>The combined local and global GPU memory used, formatted into a string</returns>
        public string GetTotalGPUMemoryUsedFormatted()
        {
            return ByteCountToFormattedString(GetTotalGPUMemoryUsed());
        }

        /// <summary>
        /// Returns the total local GPU memory used by the compute shader as bytes
        /// </summary>
        /// <returns>The total local GPU memory used as bytes</returns>
        public long GetGPUMemoryUsed()
        {
            return localGPUMemoryUsage;
        }

        /// <summary>
        /// Returns the total local GPU memory used by the compute shader, calculated and formatted into a string showing the unit
        /// </summary>
        /// <returns>The total local GPU memory used, formatted into a string</returns>
        public string GetGPUMemoryUsedFormatted()
        {
            return ByteCountToFormattedString(GetGPUMemoryUsed());
        }

        /// <summary>
        /// Returns the total global GPU memory used as bytes
        /// </summary>
        /// <returns>The total global GPU memory used as bytes</returns>
        public static long GetGlobalGPUMemoryUsed()
        {
            return globalGPUMemoryUsage;
        }

        /// <summary>
        /// Returns the total global GPU memory used, calculated and formatted into a string showing the unit
        /// </summary>
        /// <returns>The total global GPU memory used, formatted into a string</returns>
        public static string GetGlobalGPUMemoryUsedFormatted()
        {
            return ByteCountToFormattedString(GetGlobalGPUMemoryUsed());
        }

        private static string ByteCountToFormattedString(long byteCount)
        {
            double value = byteCount;
            string unit = "bytes";

            if (value > 1000)
            {
                value /= 1000d;
                unit = "KB";
            }
            if (value > 1000)
            {
                value /= 1000d;
                unit = "MB";
            }
            if (value > 1000)
            {
                value /= 1000d;
                unit = "GB";
            }
            if (value > 1000)
            {
                value /= 1000d;
                unit = "TB";
            }
            if (value > 1000)
            {
                value /= 1000d;
                unit = "PB";
            }
            if (value > 0)
            {
                return value.ToString("#.##") + " " + unit;
            }
            return value + " " + unit;
        }

        /// <summary>
        /// Sets the compute queue type
        /// </summary>
        /// <param name="computeQueueType">The compute queue type to set</param>
        public void SetComputeQueueType(ComputeQueueType computeQueueType)
        {
            this.computeQueueType = computeQueueType;
        }

        /// <summary>
        /// Returns the compute queue type
        /// </summary>
        /// <returns>The compute queue type</returns>
        public ComputeQueueType GetComputeQueueType()
        {
            return computeQueueType;
        }

        /// <summary>
        /// Sets the synchronisation stage flags
        /// </summary>
        /// <param name="synchronisationStageFlags">The synchronisation stage flags to set</param>
        public void SetSynchronisationStageFlags(SynchronisationStageFlags synchronisationStageFlags)
        {
            this.synchronisationStageFlags = synchronisationStageFlags;
        }

        /// <summary>
        /// Returns the synchronisation stage flags
        /// </summary>
        /// <returns>The synchronisation stage flags</returns>
        public SynchronisationStageFlags GetSynchronisationStageFlags()
        {
            return synchronisationStageFlags;
        }

        /// <summary>
        /// Returns the compute shader
        /// </summary>
        /// <returns>The compute shader</returns>
        public ComputeShader GetComputeShader()
        {
            return computeShader;
        }

        /// <summary>
        /// Returns whether the compute shader is currently executing
        /// </summary>
        /// <returns>Whether the compute shader is currently executing</returns>
        public bool IsExecuting()
        {
            return isExecuting;
        }

        /// <summary>
        /// Returns the amount of time it took to compute the last dispatch
        /// </summary>
        /// <returns>The amount of time it took to compute the last dispatch</returns>
        public TimeSpan GetLastComputeTime()
        {
            return lastComputeTime;
        }

        /// <summary>
        /// Disposes all buffers (local and global)
        /// </summary>
        public void DisposeLocal()
        {
            for (int i = 0; i < buffers.Count; i++)
            {
                try
                {
                    if (buffers[i].buffer != null)
                    {
                        buffers[i].buffer.Dispose();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError(e.Message);
                }
            }
            buffers.Clear();

            try
            {
                if (commandBuffer != null)
                {
                    commandBuffer.Dispose();
                    commandBuffer = null;
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }
        }

        /// <summary>
        /// Disposes all global buffers
        /// </summary>
        public static void DisposeGlobal()
        {
            for (int i = 0; i < globalBuffers.Count; i++)
            {
                try
                {
                    if (globalBuffers[i].buffer != null)
                    {
                        globalBuffers[i].buffer.Dispose();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError(e.Message);
                }
            }
            globalBuffers.Clear();
        }

        /// <summary>
        /// Disposes all local buffers
        /// </summary>
        public void Dispose()
        {
            DisposeLocal();
            DisposeGlobal();
        }
    }
}
