using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.Collections;
using UnityEngine;
using UsefulUtilities;
using UnityEngine.Rendering;

namespace GPUComputeModule
{
    /// <summary>
    /// GPUCompute provides an easy way to setup and execute GPU compute shaders (dx11+) in Unity. Features include:
    /// <para>
    /// Easily create and manage buffers/render textures, simple async or standard execution, track GPU memory usage and execution time,
    /// automatically calculate thread group sizes and buffer strides, cleanup/dispose all render textures/buffers
    /// from one function, easily resize buffers, buffer debugging to identify issues quickly- and more!
    /// </para>
    /// <para>
    /// Remember to call Dispose() when finished to prevent memory leaks on the GPU.
    /// </para>
    /// </summary>
    /// <remarks>
    /// Author(s): Aelstraz
    /// </remarks>
    public class GPUCompute : IDisposable
    {
        private ComputeQueueType computeQueueType = ComputeQueueType.Default;
        private SynchronisationStageFlags synchronisationStageFlags = SynchronisationStageFlags.ComputeProcessing;
        private ComputeShader computeShader;
        private CommandBuffer commandBuffer;
        private GraphicsFence graphicsFence;
        private List<BufferInfo> buffers = new List<BufferInfo>();
        private static List<BufferInfo> globalBuffers = new List<BufferInfo>();
        private List<RenderTexture> renderTextures = new List<RenderTexture>();
        private static List<RenderTexture> globalRenderTextures = new List<RenderTexture>();
        private bool isExecuting = false;
        private long localGPUMemoryUsage = 0;
        private static long globalGPUMemoryUsage = 0;
        private List<string> trackedMemoryObjectNames = new List<string>();
        private static List<string> trackedGlobalMemoryObjectNames = new List<string>();
        private TimeSpan lastComputeTime = TimeSpan.Zero;
        private Vector3Int threadGroupSize = new Vector3Int(1, 1, 1);

        public delegate void OnExecuteCompleteDelegate(int kernelIndex);
        public delegate void OnReadbackCompleteDelegate(AsyncGPUReadbackRequest request, string name);
        public delegate void OnGlobalReadbackCompleteDelegate(AsyncGPUReadbackRequest request, string name, params object[] parameters);
        public event OnExecuteCompleteDelegate OnExecuteComplete;
        public event OnReadbackCompleteDelegate OnReadbackComplete;
        public static event OnGlobalReadbackCompleteDelegate OnGlobalReadbackComplete;

        private enum MathOperation
        {
            ADD,
            SUBTRACT
        }

        /// <summary>
        /// Creates a new GPUCompute instance. Remember to call Dispose() before the application closes to avoid memory leaks on the GPU, or when finished to free up GPU memory
        /// </summary>
        /// <param name="computeShader">The compute shader that will be used</param>
        /// <param name="computeQueueType">Optional. Defines how the GPU will queue the execution of the compute shader</param>
        public GPUCompute(ComputeShader computeShader, ComputeQueueType computeQueueType = ComputeQueueType.Default)
        {
            this.computeShader = computeShader;
            this.computeQueueType = computeQueueType;

            commandBuffer = new CommandBuffer();
        }

        #region Execution
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
        /// <para>
        /// Note: Only available for dx12
        /// </para>
        /// </summary>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <returns>A reference to the coroutine</returns>
        public IEnumerator ExecuteAsync(int kernelIndex = 0)
        {
            if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D12)
            {
                Debug.LogError("Async execute is only available on dx12");
            }
            else if (kernelIndex < 0)
            {
                Debug.LogError("Invalid kernel index: '" + kernelIndex + "', must be 0 or greater");
            }
            else if (!isExecuting)
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
        /// <param name="length">The total number of items to be processed in your workload (e.g. length of an array)</param>
        /// <param name="shaderNumberOfThreads">The number of threads as declared in your compute shader</param>
        public void SetCalculatedThreadGroupSize(int length, int kernelIndex)
        {
            length = Mathf.Max(1, length);
            Vector3Int groupSizes = GetKernelThreadGroupSizes(kernelIndex);

            int totalNumberOfShaderThreads = groupSizes.x * groupSizes.y * groupSizes.z;
            int newGroupSize = Mathf.Max(1, Mathf.CeilToInt((float)length / (float)totalNumberOfShaderThreads));
            threadGroupSize = new Vector3Int(newGroupSize, 1, 1);
        }

        /// <summary>
        /// Automatically calculates and sets the thread group size for a two dimensional workload (e.g. a texture)
        /// </summary>
        /// <param name="width">The width of the workload</param>
        /// <param name="height">The height of the workload</param>
        /// <param name="shaderNumberOfThreads">The number of threads as declared in your compute shader</param>
        public void SetCalculatedThreadGroupSize(int width, int height, int kernelIndex)
        {
            width = Mathf.Max(1, width);
            height = Mathf.Max(1, height);
            Vector3Int groupSizes = GetKernelThreadGroupSizes(kernelIndex);

            threadGroupSize.x = Mathf.CeilToInt((float)width / groupSizes.x);
            threadGroupSize.y = Mathf.CeilToInt((float)height / groupSizes.y);
            threadGroupSize.z = groupSizes.z;
        }

        /// <summary>
        /// Automatically calculates and sets the thread group size for a three dimensional workload (e.g. a volume)
        /// </summary>
        /// <param name="width">The width of the workload</param>
        /// <param name="height">The height of the workload</param>
        /// <param name="depth">The depth of the workload</param>
        /// <param name="shaderNumberOfThreads">The number of threads as declared in your compute shader</param>
        public void SetCalculatedThreadGroupSize(int width, int height, int depth, int kernelIndex)
        {
            width = Mathf.Max(1, width);
            height = Mathf.Max(1, height);
            depth = Mathf.Max(1, depth);
            Vector3Int groupSizes = GetKernelThreadGroupSizes(kernelIndex);

            threadGroupSize.x = Mathf.CeilToInt((float)width / groupSizes.x);
            threadGroupSize.y = Mathf.CeilToInt((float)height / groupSizes.y);
            threadGroupSize.z = Mathf.CeilToInt((float)depth / groupSizes.z);
        }

        private void SetupCommandBuffer(int kernelIndex, CommandBufferExecutionFlags commandBufferExecutionFlags)
        {
            commandBuffer.Clear();
            commandBuffer.SetExecutionFlags(commandBufferExecutionFlags);
            commandBuffer.DispatchCompute(computeShader, kernelIndex, threadGroupSize.x, threadGroupSize.y, threadGroupSize.z);

            if (commandBufferExecutionFlags == CommandBufferExecutionFlags.AsyncCompute)
            {
                graphicsFence = commandBuffer.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, synchronisationStageFlags);
            }
        }

        private void ExecuteCompleted(int kernelIndex, TimeSpan computeTime)
        {
            isExecuting = false;
            lastComputeTime = computeTime;

            if (OnExecuteComplete == null || OnExecuteComplete.Target.Equals(null) || OnExecuteComplete.Method.Equals(null))
            {
                OnExecuteComplete = null;
                return;
            }

            OnExecuteComplete(kernelIndex);
        }
        #endregion

        #region Setting/Getting Variables
        /// <summary>
        /// Sets the value of an int variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetInt(string name, int value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<int>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<int>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetInt(name, value);
        }

        /// <summary>
        /// Sets the value of a float variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetFloat(string name, float value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<float>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<float>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetFloat(name, value);
        }

        /// <summary>
        /// Sets the value of a float array variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetFloatArray(string name, float[] value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                for (int i = 0; i < value.Length; i++)
                {
                    UpdateStructMemoryUsage<float>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            for (int i = 0; i < value.Length; i++)
            {
                UpdateStructMemoryUsage<float>(ref localGPUMemoryUsage, MathOperation.ADD);
            }

            computeShader.SetFloats(name, value);
        }

        /// <summary>
        /// Sets the value of a bool variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetBool(string name, bool value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<bool>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<bool>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetBool(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector4 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector4 value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetVector(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector3 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector3 value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetVector(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector2 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector2 value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetVector(name, value);
        }

        /// <summary>
        /// Sets the value of a Vector3Int variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector3Int value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetVector(name, new Vector3(value.x, value.y, value.z));
        }

        /// <summary>
        /// Sets the value of a Vector2Int variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVector(string name, Vector2Int value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetVector(name, new Vector2(value.x, value.y));
        }

        /// <summary>
        /// Sets the value of a Vector4 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetVectorArray(string name, Vector4[] value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                for (int i = 0; i < value.Length; i++)
                {
                    UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            for (int i = 0; i < value.Length; i++)
            {
                UpdateStructMemoryUsage<Vector4>(ref localGPUMemoryUsage, MathOperation.ADD);
            }

            computeShader.SetVectorArray(name, value);
        }

        /// <summary>
        /// Sets the value of a Matrix4x4 variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetMatrix(string name, Matrix4x4 value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Matrix4x4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Matrix4x4>(ref localGPUMemoryUsage, MathOperation.ADD);

            computeShader.SetMatrix(name, value);
        }

        /// <summary>
        /// Sets the value of a Matrix4x4 array variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        public void SetMatrixArray(string name, Matrix4x4[] value)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                for (int i = 0; i < value.Length; i++)
                {
                    UpdateStructMemoryUsage<Matrix4x4>(ref localGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            for (int i = 0; i < value.Length; i++)
            {
                UpdateStructMemoryUsage<Matrix4x4>(ref localGPUMemoryUsage, MathOperation.ADD);
            }

            computeShader.SetMatrixArray(name, value);
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
        /// Enables the local shader keyword in the compute shader
        /// </summary>
        /// <param name="keyword">The name of the local shader keyword to enable</param>
        public void EnableKeyword(string keyword)
        {
            computeShader.EnableKeyword(keyword);
        }

        /// <summary>
        /// Disables the local shader keyword in the compute shader
        /// </summary>
        /// <param name="keyword">The name of the local shader keyword to disable</param>
        public void DisableKeyword(string keyword)
        {
            computeShader.DisableKeyword(keyword);
        }

        /// <summary>
        /// Enables the local shader keyword in the compute shader
        /// </summary>
        /// <param name="keyword">The local shader keyword to enable</param>
        public void EnableKeyword(LocalKeyword keyword)
        {
            computeShader.EnableKeyword(keyword);
        }

        /// <summary>
        /// Disables the local shader keyword in the compute shader
        /// </summary>
        /// <param name="keyword">The local shader keyword to disable</param>
        public void DisableKeyword(LocalKeyword keyword)
        {
            computeShader.DisableKeyword(keyword);
        }

        /// <summary>
        /// Sets the value of a global int variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalInt(string name, int value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<int>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<int>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalInteger(name, value);
        }

        /// <summary>
        /// Sets the value of a global float variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalFloat(string name, float value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<float>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<float>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalFloat(name, value);
        }

        /// <summary>
        /// Sets the value of a global float array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalFloatArray(string name, float[] value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                foreach (var v in value)
                {
                    UpdateStructMemoryUsage<float>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            foreach (var v in value)
            {
                UpdateStructMemoryUsage<float>(ref globalGPUMemoryUsage, MathOperation.ADD);
            }

            Shader.SetGlobalFloatArray(name, value);
        }

        /// <summary>
        /// Sets the value of a global float array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalFloatArray(string name, List<float> value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                foreach (var v in value)
                {
                    UpdateStructMemoryUsage<float>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            foreach (var v in value)
            {
                UpdateStructMemoryUsage<float>(ref globalGPUMemoryUsage, MathOperation.ADD);
            }

            Shader.SetGlobalFloatArray(name, value);
        }

        /// <summary>
        /// Sets the value of a global Vector variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalVector(string name, Vector4 value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalVector(name, value);
        }

        /// <summary>
        /// Sets the value of a global Vector variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalVector(string name, Vector3 value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalVector(name, value);
        }

        /// <summary>
        /// Sets the value of a global Vector variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalVector(string name, Vector2 value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalVector(name, value);
        }

        /// <summary>
        /// Sets the value of a global Vector variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalVector(string name, Vector2Int value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalVector(name, (Vector2)value);
        }

        /// <summary>
        /// Sets the value of a global Vector variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalVector(string name, Vector3Int value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalVector(name, (Vector3)value);
        }

        /// <summary>
        /// Sets the value of a global Vector array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalVectorArray(string name, Vector4[] value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                foreach (var v in value)
                {
                    UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            foreach (var v in value)
            {
                UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.ADD);
            }

            Shader.SetGlobalVectorArray(name, value);
        }

        /// <summary>
        /// Sets the value of a global Vector array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalVectorArray(string name, List<Vector4> value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                foreach (var v in value)
                {
                    UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            foreach (var v in value)
            {
                UpdateStructMemoryUsage<Vector4>(ref globalGPUMemoryUsage, MathOperation.ADD);
            }

            Shader.SetGlobalVectorArray(name, value);
        }

        /// <summary>
        /// Sets the value of a global Matrix4x4 variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalMatrix(string name, Matrix4x4 value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Matrix4x4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Matrix4x4>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalMatrix(name, value);
        }

        /// <summary>
        /// Sets the value of a global Matrix4x4 array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalMatrixArray(string name, Matrix4x4[] value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                foreach (var v in value)
                {
                    UpdateStructMemoryUsage<Matrix4x4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            foreach (var v in value)
            {
                UpdateStructMemoryUsage<Matrix4x4>(ref globalGPUMemoryUsage, MathOperation.ADD);
            }

            Shader.SetGlobalMatrixArray(name, value);
        }

        /// <summary>
        /// Sets the value of a global Matrix4x4 array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalMatrixArray(string name, List<Matrix4x4> value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                foreach (var v in value)
                {
                    UpdateStructMemoryUsage<Matrix4x4>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
                }
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            foreach (var v in value)
            {
                UpdateStructMemoryUsage<Matrix4x4>(ref globalGPUMemoryUsage, MathOperation.ADD);
            }

            Shader.SetGlobalMatrixArray(name, value);
        }

        /// <summary>
        /// Sets the value of a global Color variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <param name="value">The value to set the variable</param>
        public static void SetGlobalColor(string name, Color value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateStructMemoryUsage<Color>(ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateStructMemoryUsage<Color>(ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalColor(name, value);
        }

        /// <summary>
        /// Returns the value of a global int variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static int GetGlobalInt(string name)
        {
            return Shader.GetGlobalInteger(name);
        }

        /// <summary>
        /// Returns the value of a global float variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static float GetGlobalFloat(string name)
        {
            return Shader.GetGlobalFloat(name);
        }

        /// <summary>
        /// Returns the value of a global float array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static float[] GetGlobalFloatArray(string name)
        {
            return Shader.GetGlobalFloatArray(name);
        }

        /// <summary>
        /// Returns the value of a global Vector variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static Vector4 GetGlobalVector(string name)
        {
            return Shader.GetGlobalVector(name);
        }

        /// <summary>
        /// Returns the value of a global Vector array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static Vector4[] GetGlobalVectorArray(string name)
        {
            return Shader.GetGlobalVectorArray(name);
        }

        /// <summary>
        /// Returns the value of a global Matrix4x4 variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static Matrix4x4 GetGlobalMatrix(string name)
        {
            return Shader.GetGlobalMatrix(name);
        }

        /// <summary>
        /// Returns the value of a global Matrix4x4 array variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static Matrix4x4[] GetGlobalMatrixArray(string name)
        {
            return Shader.GetGlobalMatrixArray(name);
        }

        /// <summary>
        /// Returns the value of a global Color variable
        /// </summary>
        /// <param name="name">The name of the variable</param>
        /// <returns> The value of the variable</returns>
        public static Color GetGlobalColor(string name)
        {
            return Shader.GetGlobalColor(name);
        }
        #endregion

        #region Linking Global Buffers/Textures
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
        }

        /// <summary>
        /// Links an existing global render texture to this compute shader
        /// </summary>
        /// <param name="name">The name of the keyword in the compute shader</param>
        /// <param name="globalRenderTextureName">The global name of the render texture</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void LinkGlobalRenderTexture(string name, string globalRenderTextureName, int kernelIndex = 0, int mipLevel = 0, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            LinkGlobalRenderTexture(name, globalRenderTextureName, kernelIndex, mipLevel, renderTextureSubElement);
        }

        /// <summary>
        /// Links an existing global render texture to this compute shader
        /// </summary>
        /// <param name="name">The name of the keyword in the compute shader</param>
        /// <param name="globalRenderTextureName">The global name of the render texture</param>
        /// <param name="kernelIndices">The kernel(s) to set the render texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void LinkGlobalRenderTexture(string name, string globalRenderTextureName, int[] kernelIndices, int mipLevel = 0, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            RenderTexture renderTexture = GetGlobalRenderTexture(globalRenderTextureName);

            if (renderTexture == null)
            {
                Debug.LogError("Unable to link global texture " + name + ", texture not found");
                return;
            }

            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTexture(kernelIndex, name, renderTexture, mipLevel, renderTextureSubElement);
            }
        }

        /// <summary>
        /// Links an existing global buffer to this compute shader
        /// </summary>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="globalBufferName">The global name of the buffer</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        public void LinkGlobalBuffer(string name, string globalBufferName, int kernelIndex = 0)
        {
            LinkGlobalBuffer(name, globalBufferName, new int[] { kernelIndex });
        }

        /// <summary>
        /// Links an existing global buffer to this compute shader
        /// </summary>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="globalBufferName">The global name of the buffer</param>
        /// <param name="kernelIndices">The kernel(s) to set the buffer to</param>
        public void LinkGlobalBuffer(string name, string globalBufferName, int[] kernelIndices)
        {
            BufferInfo bufferInfo = GetGlobalBuffer(globalBufferName);
            if (bufferInfo.buffer == null)
            {
                Debug.LogError("Unable to link global buffer " + name + ", buffer not found");
                return;
            }

            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetBuffer(kernelIndex, name, bufferInfo.buffer);
            }
        }
        #endregion

        #region Creating Buffers/RenderTextures
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
                return;
            }

            int stride = MemorySizes.GetStructSizeInBytes<T>();
            if (stride > 0)
            {
                ComputeBuffer computeBuffer = new ComputeBuffer(bufferLength, stride, computeBufferType, computeBufferMode);

                buffers.Add(new BufferInfo(name, computeBuffer, typeof(T), computeBufferType, computeBufferMode, kernelIndices));
                foreach (int kernelIndex in kernelIndices)
                {
                    computeShader.SetBuffer(kernelIndex, name, computeBuffer);
                }
                UpdateBufferMemoryUsage(ref localGPUMemoryUsage, stride, bufferLength, MathOperation.ADD);
            }
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
                return;
            }

            int stride = MemorySizes.GetStructSizeInBytes<T>();
            if (stride > 0)
            {
                ComputeBuffer computeBuffer = new ComputeBuffer(bufferLength, stride, computeBufferType, computeBufferMode);
                globalBuffers.Add(new BufferInfo(name, computeBuffer, typeof(T), computeBufferType, computeBufferMode, new int[0]));
                UpdateBufferMemoryUsage(ref globalGPUMemoryUsage, stride, bufferLength, MathOperation.ADD);
            }
        }

        /// <summary>
        /// Creates and sets a new render texture 
        /// </summary>
        /// <param name="name">The name of the texture to create and set in the compute shader</param>
        /// <param name="renderTextureDescriptor">The parameters to create the render texture with</param>
        /// <param name="kernelIndex">Optional. The kernel to set the texture to. Default value is 0.</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="wrapMode">The texture wrapping mode to set</param>
        /// <param name="filterMode">The texture filtering mode to set</param>
        /// <param name="anisoLevel">The texture aniso level to set</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void CreateEmptyRenderTexture(string name, RenderTextureDescriptor renderTextureDescriptor, int kernelIndex = 0, int mipLevel = 0, TextureWrapMode wrapMode = TextureWrapMode.Clamp, FilterMode filterMode = FilterMode.Bilinear, int anisoLevel = 1, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            CreateEmptyRenderTexture(name, renderTextureDescriptor, new int[] { kernelIndex }, mipLevel, wrapMode, filterMode, anisoLevel, renderTextureSubElement);
        }

        /// <summary>
        /// Creates and sets a new render texture 
        /// </summary>
        /// <param name="name">The name of the texture to create and set in the compute shader</param>
        /// <param name="renderTextureDescriptor">The parameters to create the render texture with</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="wrapMode">The texture wrapping mode to set</param>
        /// <param name="filterMode">The texture filtering mode to set</param>
        /// <param name="anisoLevel">The texture aniso level to set</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void CreateEmptyRenderTexture(string name, RenderTextureDescriptor renderTextureDescriptor, int[] kernelIndices, int mipLevel = 0, TextureWrapMode wrapMode = TextureWrapMode.Clamp, FilterMode filterMode = FilterMode.Bilinear, int anisoLevel = 1, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            if (!ListContainsRenderTexture(name, ref renderTextures))
            {
                RenderTexture texture = new RenderTexture(renderTextureDescriptor);
                texture.name = name;
                texture.wrapMode = wrapMode;
                texture.filterMode = filterMode;
                texture.anisoLevel = anisoLevel;
                texture.Create();
                renderTextures.Add(texture);

                foreach (int kernelIndex in kernelIndices)
                {
                    computeShader.SetTexture(kernelIndex, name, texture, mipLevel, renderTextureSubElement);
                }
                UpdateTextureMemoryUsage(ref texture, ref localGPUMemoryUsage, MathOperation.ADD);
            }
            else
            {
                Debug.LogWarning("Render Texture '" + name + "' has already been created");
            }
        }

        /// <summary>
        /// Creates a new un-linked global render texture
        /// </summary>
        /// <param name="name">The name of the global texture to create</param>
        /// <param name="wrapMode">The texture wrapping mode to set</param>
        /// <param name="filterMode">The texture filtering mode to set</param>
        /// <param name="anisoLevel">The texture aniso level to set</param>
        /// <param name="renderTextureDescriptor">The parameters to create the render texture with</param>
        public static void CreateEmptyGlobalRenderTexture(string name, RenderTextureDescriptor renderTextureDescriptor, TextureWrapMode wrapMode = TextureWrapMode.Clamp, FilterMode filterMode = FilterMode.Bilinear, int anisoLevel = 1)
        {
            if (!ListContainsRenderTexture(name, ref globalRenderTextures))
            {
                RenderTexture texture = new RenderTexture(renderTextureDescriptor);
                texture.name = name;
                texture.wrapMode = wrapMode;
                texture.filterMode = filterMode;
                texture.anisoLevel = anisoLevel;
                texture.Create();
                globalRenderTextures.Add(texture);
                UpdateTextureMemoryUsage(ref texture, ref globalGPUMemoryUsage, MathOperation.ADD);
            }
            else
            {
                Debug.LogWarning("Global Render Texture '" + name + "' has already been created");
            }
        }
        #endregion

        #region Setting Buffers/Textures
        private void SetBufferData<T>(string name, ref NativeArray<T> data) where T : struct
        {
            BufferInfo bufferData = GetBuffer(name);
            if (CheckIfBufferInfoIsValid<T>(name, data.Length, bufferData))
            {
                bufferData.buffer.SetData(data);
            }
        }

        private void SetBufferData<T>(string name, ref List<T> data) where T : struct
        {
            BufferInfo bufferData = GetBuffer(name);
            if (CheckIfBufferInfoIsValid<T>(name, data.Count, bufferData))
            {
                bufferData.buffer.SetData(data);
            }
        }

        private void SetBufferData<T>(string name, ref T[] data) where T : struct
        {
            BufferInfo bufferData = GetBuffer(name);
            if (CheckIfBufferInfoIsValid<T>(name, data.Length, bufferData))
            {
                bufferData.buffer.SetData(data);
            }
        }

        private static void SetGlobalBufferData<T>(string name, ref NativeArray<T> data) where T : struct
        {
            BufferInfo bufferInfo = GetGlobalBuffer(name);
            if (CheckIfBufferInfoIsValid<T>(name, data.Length, bufferInfo))
            {
                bufferInfo.buffer.SetData(data);
            }
        }

        private static void SetGlobalBufferData<T>(string name, ref List<T> data) where T : struct
        {
            BufferInfo bufferInfo = GetGlobalBuffer(name);
            if (CheckIfBufferInfoIsValid<T>(name, data.Count, bufferInfo))
            {
                bufferInfo.buffer.SetData(data);
            }
        }

        private static void SetGlobalBufferData<T>(string name, ref T[] data) where T : struct
        {
            BufferInfo bufferInfo = GetGlobalBuffer(name);
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
        /// Sets the value of a Texture2D variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTexture(string name, ref Texture2D value, int kernelIndex = 0, int mipLevel = 0)
        {
            SetTexture(name, ref value, new int[] { kernelIndex }, mipLevel);
        }

        /// <summary>
        /// Sets the value of a Texture2D variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the texture in the compute shader</param>
        /// <param name="value">The value to set</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTexture(string name, ref Texture2D value, int[] kernelIndices, int mipLevel = 0)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.ADD);

            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTexture(kernelIndex, name, value, mipLevel);
            }
        }

        /// <summary>
        /// Sets the value of a Texture3D variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTexture(string name, ref Texture3D value, int kernelIndex = 0, int mipLevel = 0)
        {
            SetTexture(name, ref value, new int[] { kernelIndex }, mipLevel);
        }

        /// <summary>
        /// Sets the value of a Texture3D variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the texture in the compute shader</param>
        /// <param name="value">The value to set</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTexture(string name, ref Texture3D value, int[] kernelIndices, int mipLevel = 0)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.ADD);

            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTexture(kernelIndex, name, value, mipLevel);
            }
        }

        /// <summary>
        /// Sets the value of a Texture2DArray variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTextureArray(string name, ref Texture2DArray value, int kernelIndex = 0, int mipLevel = 0)
        {
            SetTextureArray(name, ref value, new int[] { kernelIndex }, mipLevel);
        }

        /// <summary>
        /// Sets the value of a Texture2DArray variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the texture in the compute shader</param>
        /// <param name="value">The value to set</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        public void SetTextureArray(string name, ref Texture2DArray value, int[] kernelIndices, int mipLevel = 0)
        {
            if (trackedMemoryObjectNames.Contains(name))
            {
                UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedMemoryObjectNames.Add(name);
            }
            UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.ADD);

            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTexture(kernelIndex, name, value, mipLevel);
            }
        }

        /// <summary>
        /// Sets the value of a Render Texture variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the variable in the compute shader</param>
        /// <param name="value">The value to set the variable</param>
        /// <param name="kernelIndex">Optional. Default value is 0, which is the first kernel in the compute shader</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void SetRenderTexture(string name, ref RenderTexture value, int kernelIndex = 0, int mipLevel = 0, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            SetRenderTexture(name, ref value, new int[] { kernelIndex }, mipLevel, renderTextureSubElement);
        }

        /// <summary>
        /// Sets the value of a Render Texture variable in the compute shader
        /// </summary>
        /// <param name="name">The name of the texture in the compute shader</param>
        /// <param name="value">The value to set</param>
        /// <param name="kernelIndices">The kernel(s) to set the texture to</param>
        /// <param name="mipLevel">Optional. Default value is 0, which is the first mip level of the texture</param>
        /// <param name="renderTextureSubElement">Optional. Defines the types of data that can be encapsulated within a RenderTexture</param>
        public void SetRenderTexture(string name, ref RenderTexture value, int[] kernelIndices, int mipLevel = 0, RenderTextureSubElement renderTextureSubElement = RenderTextureSubElement.Default)
        {
            if (renderTextures.Contains(value))
            {
                UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                renderTextures.Add(value);
            }
            UpdateTextureMemoryUsage(ref value, ref localGPUMemoryUsage, MathOperation.ADD);

            foreach (int kernelIndex in kernelIndices)
            {
                computeShader.SetTexture(kernelIndex, name, value, mipLevel, renderTextureSubElement);
            }
        }

        /// <summary>
        /// Sets the value of a global Texture2D variable
        /// </summary>
        /// <param name="name">The name of the global texture</param>
        /// <param name="value">The value to set</param>
        public static void SetGlobalTexture(string name, ref Texture2D value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateTextureMemoryUsage(ref value, ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateTextureMemoryUsage(ref value, ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalTexture(name, value);
        }

        /// <summary>
        /// Sets the value of a global Texture3D variable
        /// </summary>
        /// <param name="name">The name of the global texture</param>
        /// <param name="value">The value to set</param>
        public static void SetGlobalTexture(string name, ref Texture3D value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateTextureMemoryUsage(ref value, ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateTextureMemoryUsage(ref value, ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalTexture(name, value);
        }

        /// <summary>
        /// Sets the value of a global Texture2DArray variable
        /// </summary>
        /// <param name="name">The name of the global texture array</param>
        /// <param name="value">The value to set</param>
        public static void SetGlobalTextureArray(string name, ref Texture2DArray value)
        {
            if (trackedGlobalMemoryObjectNames.Contains(name))
            {
                UpdateTextureMemoryUsage(ref value, ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
            }
            else
            {
                trackedGlobalMemoryObjectNames.Add(name);
            }
            UpdateTextureMemoryUsage(ref value, ref globalGPUMemoryUsage, MathOperation.ADD);

            Shader.SetGlobalTexture(name, value);
        }

        /// <summary>
        /// Returns the value of a global Texture variable
        /// </summary>
        /// <param name="name">The name of the global texture</param>
        /// <returns> The value of the variable</returns>
        public static Texture GetGlobalTexture(string name)
        {
            return Shader.GetGlobalTexture(name);
        }
        #endregion

        #region Retrieving Buffer/RenderTexture Data
        /// <summary>
        /// Gets the value of a render texture and copies it to a 2D texture on the CPU
        /// </summary>
        /// <param name="name">The name of the render texture</param>
        /// <param name="texture">The 2D texture to write to</param>
        /// <param name="width">Optional. Width of the data to copy. If value is not specified, the entire width of the texture is copied</param>
        /// <param name="height">Optional. Height of the data to copy. If value is not specified, the entire height of the texture is copied</param>
        /// <param name="x">Optional. Offset x position in the render texture</param>
        /// <param name="y">Optional. Offset y position in the render texture</param>
        public void GetRenderTextureData(string name, ref Texture2D texture, int width = -1, int height = -1, int x = 0, int y = 0)
        {
            RenderTexture renderTexture = GetRenderTexture(name);
            int depth = 0;
            if (TexturesMatch(renderTexture, texture, 0, TextureDimension.Tex2D, ref width, ref height, ref depth, x, y, 0))
            {
                RenderTexture previous = RenderTexture.active;
                RenderTexture.active = renderTexture;
                texture.ReadPixels(new Rect(x, y, width, height), x, y, false);
                texture.Apply(false);
                RenderTexture.active = previous;
            }
        }

        /// <summary>
        /// Gets the value of a render texture and copies it to a 3D texture on the CPU. The dimensions of the render texture must match the dimensions of the 3D texture
        /// </summary>
        /// <param name="name">The name of the render texture to get data from</param>
        /// <param name="texture">The 3D texture to copy the data to</param>
        public void GetRenderTextureData(string name, ref Texture3D texture)
        {
            RenderTexture renderTexture = GetRenderTexture(name);
            int width = 0, height = 0, depth = 0;
            if (TexturesMatch(renderTexture, texture, texture.depth, TextureDimension.Tex3D, ref width, ref height, ref depth, 0, 0, 0))
            {
                texture.CopyPixels(renderTexture);
                texture.Apply(false);
            }
        }

        /// <summary>
        /// Gets the data from a render texture and copies it to a 2D texture array on the CPU. The dimensions of the render texture must match the dimensions of the 2D texture array
        /// </summary>
        /// <param name="name">The name of the render texture to get data from</param>
        /// <param name="texture">The 2D texture array to copy the data to</param>
        public void GetRenderTextureData(string name, ref Texture2DArray texture)
        {
            RenderTexture renderTexture = GetRenderTexture(name);
            int width = 0, height = 0, depth = 0;
            if (TexturesMatch(renderTexture, texture, texture.depth, TextureDimension.Tex2DArray, ref width, ref height, ref depth, 0, 0, 0))
            {
                texture.CopyPixels(renderTexture);
                texture.Apply(false);
            }
        }

        /// <summary>
        /// Gets the value of a global render texture and copies it to a 2D texture on the CPU
        /// </summary>
        /// <param name="name">The name of the global render texture</param>
        /// <param name="texture">The 2D texture to write to</param>
        /// <param name="width">Optional. Width of the data to copy. If value is not specified, the entire width of the texture is copied</param>
        /// <param name="height">Optional. Height of the data to copy. If value is not specified, the entire height of the texture is copied</param>
        /// <param name="x">Optional. Offset x position in the render texture</param>
        /// <param name="y">Optional. Offset y position in the render texture</param>
        public static void GetGlobalRenderTextureData(string name, ref Texture2D texture, int width = -1, int height = -1, int x = 0, int y = 0)
        {
            RenderTexture renderTexture = GetGlobalRenderTexture(name);
            int depth = 0;
            if (TexturesMatch(renderTexture, texture, 0, TextureDimension.Tex2D, ref width, ref height, ref depth, x, y, 0))
            {
                RenderTexture previous = RenderTexture.active;
                RenderTexture.active = renderTexture;
                texture.ReadPixels(new Rect(x, y, width, height), x, y, false);
                texture.Apply(false);
                RenderTexture.active = previous;
            }
        }

        /// <summary>
        /// Gets the value of a global render texture and copies it to a 3D texture on the CPU. The dimensions of the render texture must match the dimensions of the 3D texture.
        /// </summary>
        /// <param name="name">The name of the global render texture to get data from</param>
        /// <param name="texture">The 3D texture to copy the data to</param>
        public static void GetGlobalRenderTextureData(string name, ref Texture3D texture)
        {
            RenderTexture renderTexture = GetGlobalRenderTexture(name);
            int width = 0, height = 0, depth = 0;
            if (TexturesMatch(renderTexture, texture, texture.depth, TextureDimension.Tex3D, ref width, ref height, ref depth, 0, 0, 0))
            {
                texture.CopyPixels(renderTexture);
                texture.Apply(false);
            }
        }

        /// <summary>
        /// Gets the value of a global render texture and copies it to a 2D texture array on the CPU. The dimensions of the render texture must match the dimensions of the 2D texture array.
        /// </summary>
        /// <param name="name">The name of the global render texture to get data from</param>
        /// <param name="texture">The 2D texture array to copy the data to</param>
        public static void GetGlobalRenderTextureData(string name, ref Texture2DArray texture)
        {
            RenderTexture renderTexture = GetGlobalRenderTexture(name);
            int width = 0, height = 0, depth = 0;
            if (TexturesMatch(renderTexture, texture, texture.depth, TextureDimension.Tex2DArray, ref width, ref height, ref depth, 0, 0, 0))
            {
                texture.CopyPixels(renderTexture);
                texture.Apply(false);
            }
        }

        /// <summary>
        /// Gets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        /// <param name="outputStartIndex">Optional. The start index of the output array</param>
        /// <param name="bufferStartIndex">Optional. The start index of the buffer</param>
        /// <param name="count">Optional. The number of elements to copy from the buffer. If count is <= 0, then all elements will be copied</param>
        public void GetBufferData<T>(string name, ref T[] output, int outputStartIndex = 0, int bufferStartIndex = 0, int count = -1) where T : struct
        {
            BufferInfo bufferData = GetBuffer(name);
            if (BuffersMatch<T>(name, bufferData, output.Length, outputStartIndex, bufferStartIndex, ref count))
            {
                bufferData.buffer.GetData(output, outputStartIndex, bufferStartIndex, count);
            }
        }

        /// <summary>
        /// Gets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        /// <param name="outputStartIndex">Optional. The start index of the output array</param>
        /// <param name="bufferStartIndex">Optional. The start index of the buffer</param>
        /// <param name="count">Optional. The number of elements to copy from the buffer. If count is <= 0, then all elements will be copied</param>
        public void GetBufferData<T>(string name, ref List<T> output, int outputStartIndex = 0, int bufferStartIndex = 0, int count = -1) where T : struct
        {
            BufferInfo bufferData = GetBuffer(name);
            if (BuffersMatch<T>(name, bufferData, output.Count, outputStartIndex, bufferStartIndex, ref count))
            {
                T[] temp = new T[output.Count];
                bufferData.buffer.GetData(temp, outputStartIndex, bufferStartIndex, count);
                for (int i = 0; i < temp.Length; i++)
                {
                    output[i] = temp[i];
                }
            }
        }

        /// <summary>
        /// Gets the value of a buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        /// <param name="outputStartIndex">Optional. The start index of the output array</param>
        /// <param name="bufferStartIndex">Optional. The start index of the buffer</param>
        /// <param name="count">Optional. The number of elements to copy from the buffer. If count is <= 0, then all elements will be copied</param>
        public void GetBufferData<T>(string name, ref NativeArray<T> output, int outputStartIndex = 0, int bufferStartIndex = 0, int count = -1) where T : struct
        {
            BufferInfo bufferData = GetBuffer(name);
            if (BuffersMatch<T>(name, bufferData, output.Length, outputStartIndex, bufferStartIndex, ref count))
            {
                T[] temp = new T[output.Length];
                bufferData.buffer.GetData(temp, outputStartIndex, bufferStartIndex, count);
                for (int i = 0; i < temp.Length; i++)
                {
                    output[i] = temp[i];
                }
            }
        }

        /// <summary>
        /// Gets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        /// <param name="outputStartIndex">Optional. The start index of the output array</param>
        /// <param name="bufferStartIndex">Optional. The start index of the buffer</param>
        /// <param name="count">Optional. The number of elements to copy from the buffer. If count is <= 0, then all elements will be copied</param>
        public static void GetGlobalBufferData<T>(string name, ref T[] output, int outputStartIndex = 0, int bufferStartIndex = 0, int count = -1) where T : struct
        {
            BufferInfo bufferData = GetGlobalBuffer(name);
            if (BuffersMatch<T>(name, bufferData, output.Length, outputStartIndex, bufferStartIndex, ref count))
            {
                bufferData.buffer.GetData(output, outputStartIndex, bufferStartIndex, count);
            }
        }

        /// <summary>
        /// Gets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        /// <param name="outputStartIndex">Optional. The start index of the output array</param>
        /// <param name="bufferStartIndex">Optional. The start index of the buffer</param>
        /// <param name="count">Optional. The number of elements to copy from the buffer. If count is <= 0, then all elements will be copied</param>
        public static void GetGlobalBufferData<T>(string name, ref List<T> output, int outputStartIndex = 0, int bufferStartIndex = 0, int count = -1) where T : struct
        {
            BufferInfo bufferData = GetGlobalBuffer(name);
            if (BuffersMatch<T>(name, bufferData, output.Count, outputStartIndex, bufferStartIndex, ref count))
            {
                T[] temp = new T[output.Count];
                bufferData.buffer.GetData(temp, outputStartIndex, bufferStartIndex, count);
                for (int i = 0; i < temp.Length; i++)
                {
                    output[i] = temp[i];
                }
            }
        }

        /// <summary>
        /// Gets the value of a global buffer in the compute shader
        /// </summary>
        /// <typeparam name="T">The struct type of the buffer</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <param name="output">The target to output the buffer data to</param>
        /// <param name="outputStartIndex">Optional. The start index of the output array</param>
        /// <param name="bufferStartIndex">Optional. The start index of the buffer</param>
        /// <param name="count">Optional. The number of elements to copy from the buffer. If count is <= 0, then all elements will be copied</param>
        public static void GetGlobalBufferData<T>(string name, ref NativeArray<T> output, int outputStartIndex = 0, int bufferStartIndex = 0, int count = -1) where T : struct
        {
            BufferInfo bufferData = GetGlobalBuffer(name);
            if (BuffersMatch<T>(name, bufferData, output.Length, outputStartIndex, bufferStartIndex, ref count))
            {
                T[] temp = new T[output.Length];
                bufferData.buffer.GetData(temp, outputStartIndex, bufferStartIndex, count);
                for (int i = 0; i < temp.Length; i++)
                {
                    output[i] = temp[i];
                }
            }
        }

        /// <summary>
        /// Finds and returns the BufferInfo that has the same input name
        /// </summary>
        /// <param name="name">The name of the buffer</param>
        /// <returns>The BufferInfo that matches the input name, or a BufferInfo containing null values</returns>
        public BufferInfo GetBuffer(string name)
        {
            BufferInfo bufferData = new BufferInfo(null, null, null, ComputeBufferType.Default, ComputeBufferMode.Dynamic, new int[0]);
            foreach (BufferInfo buffer in buffers)
            {
                if (buffer.name == name)
                {
                    bufferData = buffer;
                    break;
                }
            }
            return bufferData;
        }

        /// <summary>
        /// Finds and returns the RenderTexture that has the same input name
        /// </summary>
        /// <param name="name">The name of the render texture</param>
        /// <returns>The RenderTexture that matches the input name, or null</returns>
        public RenderTexture GetRenderTexture(string name)
        {
            RenderTexture renderTexture = null;
            foreach (RenderTexture texture in renderTextures)
            {
                if (texture.name == name)
                {
                    renderTexture = texture;
                    break;
                }
            }
            return renderTexture;
        }

        // <summary>
        /// Finds and returns the global BufferInfo that has the same input name
        /// </summary>
        /// <param name="name">The name of the global buffer</param>
        /// <returns>The global BufferInfo that matches the input name, or a BufferInfo containing null values</returns>
        public static BufferInfo GetGlobalBuffer(string name)
        {
            BufferInfo bufferData = new BufferInfo(null, null, null, ComputeBufferType.Default, ComputeBufferMode.Dynamic, new int[0]);
            foreach (BufferInfo buffer in globalBuffers)
            {
                if (buffer.name == name)
                {
                    bufferData = buffer;
                    break;
                }
            }
            return bufferData;
        }

        /// <summary>
        /// Finds and returns the global RenderTexture that has the same input name
        /// </summary>
        /// <param name="name">The name of the global render texture</param>
        /// <returns>The global RenderTexture that matches the input name, or null</returns>
        public static RenderTexture GetGlobalRenderTexture(string name)
        {
            RenderTexture renderTexture = null;
            foreach (RenderTexture texture in globalRenderTextures)
            {
                if (texture.name == name)
                {
                    renderTexture = texture;
                    break;
                }
            }
            return renderTexture;
        }
        #endregion

        #region Async Data Retrieval
        /// <summary>
        /// Asynchronously gets the wanted values of a buffer in the compute shader using AsyncGPUReadbackRequest. This is more efficient than GetBufferData, but requires a callback to retrieve the data once the request is complete.
        /// </summary>
        /// <param name="bufferName">The name of the buffer in the compute shader</param>
        /// <param name="length">Optional. The number of elements in the buffer to retrieve. If not specified, the entire buffer will be retrieved</param>
        /// <param name="startIndex">Optional. The index to begin retrieving data from</param>
        /// <returns>A coroutine that can be started to asynchronously read back the data from a buffer</returns>
        public IEnumerator GetBufferDataAsync(string bufferName, int length = -1, int startIndex = 0)
        {
            BufferInfo bufferInfo = GetBuffer(bufferName);
            if (bufferInfo.buffer == null)
            {
                Debug.LogError("Error getting buffer data, unable to find buffer '" + bufferName + "'");
            }
            else if (startIndex < 0 || startIndex >= bufferInfo.GetCount())
            {
                Debug.LogError($"Error getting buffer data, invalid startIndex '{startIndex}' must be between 0 and  '{bufferInfo.GetCount() - 1}'");
            }
            else
            {
                if (length <= 0)
                {
                    length = bufferInfo.GetCount();
                }

                if (startIndex + length > bufferInfo.GetCount())
                {
                    Debug.LogError($"Error getting buffer data, length '{length}' exceeds the bounds of the buffer. Length must be between 0 and '{bufferInfo.GetCount() - startIndex}'");
                }
                else
                {
                    int structSize = MemorySizes.GetStructSizeInBytes(bufferInfo.dataType);
                    AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(bufferInfo.buffer, length * structSize, startIndex * structSize);
                    yield return new WaitWhile(() => !request.done);

                    if (request.hasError)
                    {
                        Debug.LogError("Error retrieving buffer data, AsyncGPUReadbackRequest error");
                    }
                    else
                    {
                        if (OnReadbackComplete == null || OnReadbackComplete.Target == null || OnReadbackComplete.Method == null)
                        {
                            OnReadbackComplete = null;
                        }
                        else
                        {
                            OnReadbackComplete(request, bufferName);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Asynchronously gets the wanted values of a global buffer in the compute shader using AsyncGPUReadbackRequest. This is more efficient than GetGlobalBufferData, but requires a callback to retrieve the data once the request is complete.
        /// </summary>
        /// <param name="bufferName">The name of the global buffer in the compute shader</param>
        /// <param name="length">Optional. The number of elements in the buffer to retrieve. If not specified, the entire buffer will be retrieved</param>
        /// <param name="startIndex">Optional. The index to begin retrieving data from</param>
        /// <param name="parameters">Optional. Additional parameters that can be passed through for your own use</param>
        /// <returns>A coroutine that can be started to asynchronously read back the data from a buffer</returns>
        public static IEnumerator GetGlobalBufferDataAsync(string bufferName, int length = -1, int startIndex = 0, params object[] parameters)
        {
            BufferInfo bufferInfo = GetGlobalBuffer(bufferName);
            if (bufferInfo.buffer == null)
            {
                Debug.LogError("Error getting global buffer data, unable to find global buffer '" + bufferName + "'");
            }
            else if (startIndex < 0 || startIndex >= bufferInfo.GetCount())
            {
                Debug.LogError($"Error getting global buffer data, invalid startIndex '{startIndex}' must be between 0 and  '{bufferInfo.GetCount() - 1}'");
            }
            else
            {
                if (length <= 0)
                {
                    length = bufferInfo.GetCount();
                }

                if (startIndex + length > bufferInfo.GetCount())
                {
                    Debug.LogError($"Error getting global buffer data, length '{length}' exceeds the bounds of the global buffer. Length must be between 0 and '{bufferInfo.GetCount() - startIndex}'");
                }
                else
                {
                    int structSize = MemorySizes.GetStructSizeInBytes(bufferInfo.dataType);
                    AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(bufferInfo.buffer, length * structSize, startIndex * structSize);
                    yield return new WaitWhile(() => !request.done);

                    if (request.hasError)
                    {
                        Debug.LogError("Error retrieving global buffer data, AsyncGPUReadbackRequest error");
                    }
                    else
                    {
                        if (OnGlobalReadbackComplete == null || OnGlobalReadbackComplete.Target == null || OnGlobalReadbackComplete.Method == null)
                        {
                            OnGlobalReadbackComplete = null;
                        }
                        else
                        {
                            OnGlobalReadbackComplete(request, bufferName, parameters);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Asynchronously gets the wanted values of a render texture in the compute shader using AsyncGPUReadbackRequest. This is more efficient than GetRenderTextureData, but requires a callback to retrieve the data once the request is complete.
        /// </summary>
        /// <param name="textureName">The name of the render texture in the compute shader</param>
        /// <param name="width">The x length to retrieve pixels</param>
        /// <param name="height">The y length to retrieve pixels</param>
        /// <param name="depth">The z length to retrieve pixels</param>
        /// <param name="mipIndex">Optional. The mip level to retrieve pixels</param>
        /// <param name="x">Optional. The starting x coordinate to retrieve pixels</param>
        /// <param name="y">Optional. The starting y coordinate to retrieve pixels</param>
        /// <param name="z">Optional. The starting z coordinate to retrieve pixels</param>
        /// <returns>A coroutine that can be started to asynchronously read back the pixels from a global render texture</returns>
        public IEnumerator GetRenderTextureDataAsync(string textureName, int width = -1, int height = -1, int depth = -1, int mipIndex = 0, int x = 0, int y = 0, int z = 0)
        {
            RenderTexture renderTexture = GetRenderTexture(textureName);
            if (renderTexture == null)
            {
                Debug.LogError($"Error getting texture data, unable to find render texture '{textureName}'");
            }
            else
            {
                if (width <= 0)
                {
                    width = renderTexture.width;
                }
                if (height <= 0)
                {
                    height = renderTexture.height;
                }
                if (depth <= 0)
                {
                    depth = renderTexture.volumeDepth;
                }

                if (x + width > renderTexture.width || y + height > renderTexture.height || z + depth > renderTexture.volumeDepth)
                {
                    Debug.LogError($"Error getting texture data, coordinates out of bounds for render texture '{textureName}'. "
                        + $"Area: ({width}, {height}, {depth}). Coordinates: ({x}, {y}, {z}). Texture dimensions: ({renderTexture.width}, {renderTexture.height}, {renderTexture.volumeDepth})");
                }
                else if (x >= renderTexture.width || y >= renderTexture.height || z >= renderTexture.volumeDepth)
                {
                    Debug.LogError($"Error getting texture data, coordinates out of bounds for render texture '{textureName}'. "
                        + $"Area: ({width}, {height}, {depth}). Coordinates: ({x}, {y}, {z}). Texture dimensions: ({renderTexture.width}, {renderTexture.height}, {renderTexture.volumeDepth})");
                }
                else
                {
                    AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(renderTexture, mipIndex, x, width, y, height, z, depth);
                    yield return new WaitWhile(() => !request.done);

                    if (request.hasError)
                    {
                        Debug.LogError("Error retrieving texture data, AsyncGPUReadbackRequest error");
                    }
                    else
                    {
                        if (OnReadbackComplete == null || OnReadbackComplete.Target == null || OnReadbackComplete.Method == null)
                        {
                            OnReadbackComplete = null;
                        }
                        else
                        {
                            OnReadbackComplete(request, textureName);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Asynchronously gets the wanted values of a global render texture in the compute shader using AsyncGPUReadbackRequest. This is more efficient than GetGlobalRenderTextureData, but requires a callback to retrieve the data once the request is complete.
        /// </summary>
        /// <param name="textureName">The name of the global render texture in the compute shader</param>
        /// <param name="width">Optional. The x length to retrieve pixels. If not specified, all pixels in the x dimension will be retrieved</param>
        /// <param name="height">Optional. The y length to retrieve pixels. If not specified, all pixels in the y dimension will be retrieved</param>
        /// <param name="depth">Optional. The z length to retrieve pixels. If not specified, all pixels in the z dimension will be retrieved</param>
        /// <param name="mipIndex">Optional. The mip level to retrieve pixels</param>
        /// <param name="x">Optional. The starting x coordinate to retrieve pixels. If not specified, the starting x coordinate will be 0</param>
        /// <param name="y">Optional. The starting y coordinate to retrieve pixels. If not specified, the starting y coordinate will be 0</param>
        /// <param name="z">Optional. The starting z coordinate to retrieve pixels. If not specified, the starting z coordinate will be 0</param>
        /// <param name="parameters">Optional. Additional parameters that can be passed through for your own use</param>
        /// <returns>A coroutine that can be started to asynchronously read back the pixels from a global render texture</returns>
        public static IEnumerator GetGlobalRenderTextureDataAsync(string textureName, int width = -1, int height = -1, int depth = -1, int mipIndex = 0, int x = 0, int y = 0, int z = 0, params object[] parameters)
        {
            RenderTexture renderTexture = GetGlobalRenderTexture(textureName);
            if (renderTexture == null)
            {
                Debug.LogError($"Error getting global texture data, unable to find global render texture '{textureName}'");
            }
            else
            {
                if (width <= 0)
                {
                    width = renderTexture.width;
                }
                if (height <= 0)
                {
                    height = renderTexture.height;
                }
                if (depth <= 0)
                {
                    depth = renderTexture.volumeDepth;
                }

                if (x + width > renderTexture.width || y + height > renderTexture.height || z + depth > renderTexture.volumeDepth)
                {
                    Debug.LogError($"Error getting global texture data, coordinates out of bounds for global render texture '{textureName}'. "
                        + $"Area: ({width}, {height}, {depth}). Coordinates: ({x}, {y}, {z}). Global texture dimensions: ({renderTexture.width}, {renderTexture.height}, {renderTexture.volumeDepth})");
                }
                else if (x >= renderTexture.width || y >= renderTexture.height || z >= renderTexture.volumeDepth)
                {
                    Debug.LogError($"Error getting global texture data, coordinates out of bounds for global render texture '{textureName}'. "
                        + $"Area: ({width}, {height}, {depth}). Coordinates: ({x}, {y}, {z}). Global texture dimensions: ({renderTexture.width}, {renderTexture.height}, {renderTexture.volumeDepth})");
                }
                else
                {
                    AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(renderTexture, mipIndex, x, width, y, height, z, depth);
                    yield return new WaitWhile(() => !request.done);

                    if (request.hasError)
                    {
                        Debug.LogError("Error retrieving texture data, AsyncGPUReadbackRequest error");
                    }
                    else
                    {
                        if (OnGlobalReadbackComplete == null || OnGlobalReadbackComplete.Target == null || OnGlobalReadbackComplete.Method == null)
                        {
                            OnGlobalReadbackComplete = null;
                        }
                        else
                        {
                            OnGlobalReadbackComplete(request, textureName, parameters);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Reads back the data from an AsyncGPUReadbackRequest into a NativeArray on the CPU.
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="request">The AsyncGPUReadbackRequest to read back data from</param>
        /// <param name="data">The native array to read the data into</param>
        public static void ReadbackRequestToNativeArray<T>(ref AsyncGPUReadbackRequest request, ref NativeArray<T> data) where T : struct
        {
            if (request.width != data.Length * MemorySizes.GetStructSizeInBytes<T>() || request.height != 1)
            {
                Debug.LogError("Unable to readback to native array, request and data dimensions do not match");
                return;
            }
            data = request.GetData<T>();
        }

        /// <summary>
        /// Reads back the data from an AsyncGPUReadbackRequest into a Array on the CPU.
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="request">The AsyncGPUReadbackRequest to read back data from</param>
        /// <param name="data">The array to read the data into</param>
        public static void ReadbackRequestToArray<T>(ref AsyncGPUReadbackRequest request, ref T[] data) where T : struct
        {
            if (request.width != data.Length * MemorySizes.GetStructSizeInBytes<T>() || request.height != 1)
            {
                Debug.LogError("Unable to readback to native array, request and data dimensions do not match");
                return;
            }
            data = request.GetData<T>().ToArray();
        }

        /// <summary>
        /// Reads back the data from an AsyncGPUReadbackRequest into a List on the CPU.
        /// </summary>
        /// <typeparam name="T">The struct type of the data</typeparam>
        /// <param name="request">The AsyncGPUReadbackRequest to read back data from</param>
        /// <param name="data">The list to read the data into</param>
        public static void ReadbackRequestToList<T>(ref AsyncGPUReadbackRequest request, ref List<T> data) where T : struct
        {
            if (request.width != data.Count * MemorySizes.GetStructSizeInBytes<T>() || request.height != 1)
            {
                Debug.LogError("Unable to readback to native array, request and data dimensions do not match");
                return;
            }
            data = request.GetData<T>().ToList();
        }

        /// <summary>
        /// Reads back the data from an AsyncGPUReadbackRequest into a Texture2D on the CPU.
        /// </summary>
        /// <param name="request">The AsyncGPUReadbackRequest to read back data from</param>
        /// <param name="texture">The texture2D to read the data into</param>
        /// <param name="mipLevel">Optional. The mip level to retrieve pixels</param>
        public static void ReadbackRequestToTexture2D(ref AsyncGPUReadbackRequest request, ref Texture2D texture, int mipLevel = 0)
        {
            if (request.width != texture.width || request.height != texture.height)
            {
                Debug.LogError("Unable to readback to texture2D, request and texture dimensions do not match");
                return;
            }

            texture.SetPixelData(request.GetData<Color>(), mipLevel);
            texture.Apply(false);
        }

        /// <summary>
        /// Reads back the data from an AsyncGPUReadbackRequest into a Texture3D on the CPU.
        /// </summary>
        /// <param name="request">The AsyncGPUReadbackRequest to read back data from</param>
        /// <param name="texture">The texture3D to read the data into</param>
        /// <param name="mipLevel">Optional. The mip level to retrieve pixels</param>
        public static void ReadbackRequestToTexture3D(ref AsyncGPUReadbackRequest request, ref Texture3D texture, int mipLevel = 0)
        {
            if (request.width != texture.width || request.height != texture.height || request.depth != texture.depth)
            {
                Debug.LogError($"Unable to readback to texture3D, request and texture dimensions do not match. "
                    + $"Request dimensions: ({request.width}, {request.height}, {request.depth}). Texture dimensions: ({texture.width}, {texture.height}, {texture.depth})");
                return;
            }

            Color[] data = new Color[request.width * request.height * request.layerCount];
            NativeArray<Color> slice;
            int offset = 0;

            for (int i = 0; i < request.layerCount; i++)
            {
                slice = request.GetData<Color>(i);
                for (int x = 0; x < slice.Length; x++)
                {
                    data[offset + x] = slice[x];
                }
                offset += slice.Length;
            }

            texture.SetPixelData(data, mipLevel);
            texture.Apply(false);
        }

        /// <summary>
        /// Reads back the data from an AsyncGPUReadbackRequest into a Texture2DArray on the CPU.
        /// </summary>
        /// <param name="request">The AsyncGPUReadbackRequest to read back data from</param>
        /// <param name="texture">The texture3D to read the data into</param>
        /// <param name="mipLevel">Optional. The mip level to retrieve pixels</param>
        public static void ReadbackRequestToTexture2DArray(ref AsyncGPUReadbackRequest request, ref Texture2DArray texture, int mipLevel = 0)
        {
            if (request.width != texture.width || request.height != texture.height || request.depth != texture.depth)
            {
                Debug.LogError($"Unable to readback to Texture2DArray, request and texture dimensions do not match. "
                    + $"Request dimensions: ({request.width}, {request.height}, {request.depth}). Texture dimensions: ({texture.width}, {texture.height}, {texture.depth})");
                return;
            }

            for (int i = 0; i < request.layerCount; i++)
            {
                texture.SetPixelData(request.GetData<Color>(i), i, mipLevel);
            }
            texture.Apply(false);
        }
        #endregion

        #region Buffer Resizing
        /// <summary>
        /// Changes the length of a buffer (all data in the buffer is lost)
        /// </summary>
        /// <param name="name">The name of the buffer in the compute shader</param>
        /// <param name="length">The desired length of the buffer</param>
        public void ResizeBuffer(string name, int length)
        {
            int index = IndexOfBufferInList(name, ref buffers);

            if (index == -1)
            {
                Debug.LogWarning("Buffer " + name + " has not been created yet");
                return;
            }

            BufferInfo bufferInfo = GetBuffer(name);
            int stride = bufferInfo.GetStride();
            UpdateBufferMemoryUsage(ref localGPUMemoryUsage, stride, bufferInfo.GetCount(), MathOperation.SUBTRACT);
            bufferInfo.buffer.Dispose();
            bufferInfo.buffer = null;
            buffers.RemoveAt(index);

            ComputeBuffer newBuffer = new ComputeBuffer(length, stride, bufferInfo.computeBufferType, bufferInfo.computeBufferMode);
            buffers.Add(new BufferInfo(name, newBuffer, bufferInfo.dataType, bufferInfo.computeBufferType, bufferInfo.computeBufferMode, bufferInfo.kernelIndices));
            foreach (int kernelIndex in bufferInfo.kernelIndices)
            {
                computeShader.SetBuffer(kernelIndex, name, newBuffer);
            }
            UpdateBufferMemoryUsage(ref localGPUMemoryUsage, stride, length, MathOperation.ADD);
        }

        /// <summary>
        /// Changes the length of a global buffer (all data in the global buffer is lost)
        /// </summary>
        /// <param name="name">The name of the global buffer in the compute shader</param>
        /// <param name="length">The desired length of the global buffer</param>
        public static void ResizeGlobalBuffer(string name, int length)
        {
            int index = IndexOfBufferInList(name, ref globalBuffers);

            if (index == -1)
            {
                Debug.LogWarning("Buffer " + name + " has not been created yet");
                return;
            }

            BufferInfo bufferInfo = GetGlobalBuffer(name);
            int stride = bufferInfo.GetStride();
            UpdateBufferMemoryUsage(ref globalGPUMemoryUsage, stride, bufferInfo.GetCount(), MathOperation.SUBTRACT);
            bufferInfo.buffer.Dispose();
            globalBuffers.RemoveAt(index);

            ComputeBuffer newBuffer = new ComputeBuffer(length, stride, bufferInfo.computeBufferType, bufferInfo.computeBufferMode);
            globalBuffers.Add(new BufferInfo(name, newBuffer, bufferInfo.dataType, bufferInfo.computeBufferType, bufferInfo.computeBufferMode, bufferInfo.kernelIndices));
            UpdateBufferMemoryUsage(ref globalGPUMemoryUsage, stride, length, MathOperation.ADD);
        }
        #endregion

        #region Utility Functions
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

        private static bool ListContainsRenderTexture(string name, ref List<RenderTexture> renderTextures)
        {
            bool contains = false;
            foreach (RenderTexture texture in renderTextures)
            {
                if (texture.name == name)
                {
                    contains = true;
                    break;
                }
            }
            return contains;
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

        private static int IndexOfTextureInList(string name, ref List<RenderTexture> renderTextureList)
        {
            int index = -1;
            for (int i = 0; i < renderTextureList.Count; i++)
            {
                if (renderTextureList[i].name == name)
                {
                    index = i;
                    break;
                }
            }
            return index;
        }

        private static bool TexturesMatch(RenderTexture renderTexture, Texture texture, int textureDepth, TextureDimension textureDimension, ref int width, ref int height, ref int depth, int x, int y, int z)
        {
            bool isValid = true;
            if (renderTexture == null)
            {
                Debug.LogError("Unable to get global render texture data, global render texture is null");
                isValid = false;
            }
            else if (texture == null)
            {
                Debug.LogError("Unable to get global render texture data, supplied texture is null");
                isValid = false;
            }
            else if (!renderTexture.isReadable)
            {
                Debug.LogError($"Unable to get global render texture data, global render texture '{renderTexture.name}' is not readable");
                isValid = false;
            }
            else if (!texture.isReadable)
            {
                Debug.LogError($"Unable to get global render texture data, supplied texture '{texture.name}' is not readable");
                isValid = false;
            }
            else if (texture.dimension != textureDimension)
            {
                Debug.LogError($"Unable to get global render texture data, supplied texture '{texture.name}' has a dimension of {texture.dimension}, expected dimension of {textureDimension}");
                isValid = false;
            }
            else if (renderTexture.dimension != textureDimension)
            {
                Debug.LogError($"Unable to get global render texture data, global render texture '{renderTexture.name}' has a dimension of {renderTexture.dimension}, expected dimension of {textureDimension}");
                isValid = false;
            }
            else
            {
                if (width <= 0)
                {
                    width = texture.width;
                }
                if (height <= 0)
                {
                    height = texture.height;
                }
                if (depth <= 0)
                {
                    depth = textureDepth;
                }

                if (width + x > renderTexture.width || height + y > renderTexture.height || depth + z > renderTexture.depth)
                {
                    Debug.LogError($"Unable to get global render texture data, global render texture '{renderTexture.name}' doesn't have enough space to hold the global render texture data");
                    isValid = false;
                }
                else if (width + x > texture.width || height + y > texture.height || depth + z > textureDepth)
                {
                    Debug.LogError($"Unable to get global render texture data, the supplied texture '{texture.name}' doesn't have enough space to hold the global render texture data");
                    isValid = false;
                }
            }
            return isValid;
        }

        private static bool BuffersMatch<T>(string name, BufferInfo bufferData, int outputLength, int outputStartIndex, int bufferStartIndex, ref int count)
        {
            bool isValid = true;
            if (string.IsNullOrEmpty(bufferData.name) || bufferData.buffer == null)
            {
                Debug.LogError($"Unable to get buffer data '{name}', buffer not found");
                isValid = false;
            }
            else if (bufferData.dataType != typeof(T))
            {
                Debug.LogError($"Unable to get buffer data '{name}', supplied type is incorrect. Buffer type is '{bufferData.dataType}', Supplied type is '{typeof(T)}'");
                isValid = false;
            }
            else if (bufferData.GetCount() <= 0)
            {
                Debug.LogError($"Unable to get buffer data '{name}', buffer data count is zero");
                isValid = false;
            }
            else if (outputLength <= 0)
            {
                Debug.LogError($"Unable to get buffer data '{name}', supplied array data count is zero");
                isValid = false;
            }
            else if (bufferData.GetCount() != outputLength)
            {
                Debug.LogError($"Unable to get buffer data '{name}', supplied array length is incorrect. Expected length '{bufferData.GetCount()}', Supplied length '{outputLength}'");
                isValid = false;
            }
            else if (outputStartIndex < 0 || outputStartIndex >= outputLength)
            {
                Debug.LogError($"Unable to get buffer data '{name}', supplied start index is out of range. Expected range '0 - {outputLength - 1}', Supplied start index '{outputStartIndex}'");
                isValid = false;
            }
            else if (bufferStartIndex < 0 || bufferStartIndex >= bufferData.GetCount())
            {
                Debug.LogError($"Unable to get buffer data '{name}', supplied start index is out of range. Expected range '0 - {bufferData.GetCount() - 1}', Supplied start index '{bufferStartIndex}'");
                isValid = false;
            }

            if (count <= 0)
            {
                count = outputLength;
            }

            return isValid;
        }
        #endregion

        #region Memory Tracking
        private static void UpdateStructMemoryUsage<T>(ref long totalMemoryUsage, MathOperation operation) where T : struct
        {
            long value = MemorySizes.GetStructSizeInBytes<T>();
            if (operation == MathOperation.ADD)
            {
                totalMemoryUsage += value;
            }
            else
            {
                totalMemoryUsage -= value;
            }
        }

        private static void UpdateBufferMemoryUsage(ref long totalMemoryUsage, int stride, long length, MathOperation operation)
        {
            long value = stride * length;
            if (operation == MathOperation.ADD)
            {
                totalMemoryUsage += value;
            }
            else
            {
                totalMemoryUsage -= value;
            }
        }

        private static void UpdateTextureMemoryUsage(ref Texture2DArray texture, ref long memoryUsage, MathOperation operation)
        {
            long value = texture.width * texture.height * texture.depth * MemorySizes.GetBitsPerPixel(texture.format) / 8;

            if (operation == MathOperation.ADD)
            {
                memoryUsage += value;
            }
            else if (operation == MathOperation.SUBTRACT)
            {
                memoryUsage -= value;
            }
        }

        private static void UpdateTextureMemoryUsage(ref Texture2D texture, ref long memoryUsage, MathOperation operation)
        {
            long value = texture.width * texture.height * MemorySizes.GetBitsPerPixel(texture.format) / 8;

            if (operation == MathOperation.ADD)
            {
                memoryUsage += value;
            }
            else if (operation == MathOperation.SUBTRACT)
            {
                memoryUsage -= value;
            }
        }

        private static void UpdateTextureMemoryUsage(ref Texture3D texture, ref long memoryUsage, MathOperation operation)
        {
            long value = texture.width * texture.height * texture.depth * MemorySizes.GetBitsPerPixel(texture.format) / 8;

            if (operation == MathOperation.ADD)
            {
                memoryUsage += value;
            }
            else if (operation == MathOperation.SUBTRACT)
            {
                memoryUsage -= value;
            }
        }

        private static void UpdateTextureMemoryUsage(ref RenderTexture texture, ref long memoryUsage, MathOperation operation)
        {
            long value = texture.width * texture.height * texture.volumeDepth * MemorySizes.GetBitsPerPixel(texture.format, texture.depth) / 8;

            if (operation == MathOperation.ADD)
            {
                memoryUsage += value;
            }
            else if (operation == MathOperation.SUBTRACT)
            {
                memoryUsage -= value;
            }
        }

        /// <summary>
        /// Returns the total local GPU memory used by the compute shader as bytes
        /// </summary>
        /// <returns>The total local GPU memory used as bytes</returns>
        public long GetLocalGPUMemoryUsed()
        {
            return localGPUMemoryUsage;
        }

        /// <summary>
        /// Returns the total local GPU memory used by the compute shader, calculated and formatted into a string showing the unit
        /// </summary>
        /// <returns>The total local GPU memory used, formatted into a string</returns>
        public string GetLocalGPUMemoryUsedFormatted()
        {
            return ByteCountToFormattedString(GetLocalGPUMemoryUsed());
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

        /// <summary>
        /// Converts the input byte count to its relevant unit (e.g. 1000 bytes = "1 KB")
        /// </summary>
        /// <param name="byteCount"></param>
        /// <returns>A formatted string containing the converted value followed by its unit</returns>
        public static string ByteCountToFormattedString(long byteCount)
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
        #endregion

        #region Debugging
        /// <summary>
        /// Writes the buffer from the GPU to the CPU, and returns a string containing all of the buffer's information and all contained values-
        /// and should only be used for debugging purposes
        /// </summary>
        /// <typeparam name="T">The data type of the buffer</typeparam>
        /// <param name="name">The name of the buffer</param>
        /// <returns>A string containing all of the buffer's information and all contained values</returns>
        public string DebugBuffer<T>(string name)
        {
            StringBuilder sb = new StringBuilder();
            BufferInfo bufferInfo = GetBuffer(name);

            if (bufferInfo.dataType != typeof(T))
            {
                sb.AppendLine("Error: Supplied data type doesn't match the buffer's data type: '" + bufferInfo.dataType.ToString() + "'.");
            }
            else if (bufferInfo.name != null)
            {
                sb.Append(bufferInfo.GetDebugString());
                sb.AppendLine("\nValues:");

                int count;
                if (bufferInfo.computeBufferType == ComputeBufferType.Append || bufferInfo.computeBufferType == ComputeBufferType.Counter)
                {
                    count = Mathf.Min(bufferInfo.GetAppendCountFromGPU(), bufferInfo.GetCount());
                }
                else
                {
                    count = bufferInfo.GetCount();
                }

                T[] values = new T[count];
                bufferInfo.buffer.GetData(values);

                for (int i = 0; i < values.Length; i++)
                {
                    sb.AppendLine("Index " + i + ": " + values[i]);
                }
            }
            else
            {
                sb.AppendLine("Error: Buffer '" + name + "' not found.");
            }
            return sb.ToString();
        }

        /// <summary>
        /// Writes the global buffer from the GPU to the CPU, and returns a string containing all of the global buffer's information and all contained values-
        /// and should only be used for debugging purposes
        /// </summary>
        /// <typeparam name="T">The data type of the global buffer</typeparam>
        /// <param name="name">The name of the global buffer</param>
        /// <returns>A string containing all of the global buffer's information and all contained values</returns>
        public static string DebugGlobalBuffer<T>(string name)
        {
            StringBuilder sb = new StringBuilder();
            BufferInfo bufferInfo = GetGlobalBuffer(name);

            if (bufferInfo.dataType != typeof(T))
            {
                sb.AppendLine("Error: Supplied data type doesn't match the global buffer's data type: '" + bufferInfo.dataType.ToString() + "'.");
            }
            else if (bufferInfo.name != null)
            {
                sb.Append("Global " + bufferInfo.GetDebugString());
                sb.AppendLine("\nValues:");

                int count;
                if (bufferInfo.computeBufferType == ComputeBufferType.Append || bufferInfo.computeBufferType == ComputeBufferType.Counter)
                {
                    count = bufferInfo.GetAppendCountFromGPU();
                }
                else
                {
                    count = bufferInfo.GetCount();
                }

                T[] values = new T[count];
                bufferInfo.buffer.GetData(values);

                for (int i = 0; i < values.Length; i++)
                {
                    sb.AppendLine("Index " + i + ": " + values[i]);
                }
            }
            else
            {
                sb.AppendLine("Error: Buffer '" + name + "' not found.");
            }
            return sb.ToString();
        }
        #endregion

        #region Getters/Setters
        /// <summary>
        /// Gets all currently enabled local shader keywords
        /// </summary>
        /// <returns>An array containing all enabled keywords</returns>
        public LocalKeyword[] GetEnabledKeywords()
        {
            return computeShader.enabledKeywords;
        }

        /// <summary>
        /// Checks if the kernel exists
        /// </summary>
        /// <param name="kernelName">The name of the kernel to check</param>
        /// <returns>True if the kernel exists, otherwise false</returns>
        public int FindKernel(string kernelName)
        {
            return computeShader.FindKernel(kernelName);
        }

        /// <summary>
        /// Returns the kernels thread group sizes for the given kernel index
        /// </summary>
        /// <param name="kernelIndex">The index of the kernel to check</param>
        /// <returns>A Vector3Int containing the group sizes across the x, y and z dimensions </returns>
        public Vector3Int GetKernelThreadGroupSizes(int kernelIndex)
        {
            uint x, y, z;
            computeShader.GetKernelThreadGroupSizes(kernelIndex, out x, out y, out z);
            return new Vector3Int((int)x, (int)y, (int)z);
        }

        /// <summary>
        /// Checks if the kernel exists in the compute shader
        /// </summary>
        /// <param name="kernelName">The name of the kernel to check</param>
        /// <returns>True if the kernel exists, otherwise false</returns>
        public bool HasKernel(string kernelName)
        {
            return computeShader.HasKernel(kernelName);
        }

        /// <summary>
        /// Checks if the keyword is currently enabled
        /// </summary>
        /// <param name="keyword">The keyword to check</param>
        /// <returns>True if the keyword is enabled, otherwise false</returns>
        public bool IsKeywordEnabled(string keyword)
        {
            return computeShader.IsKeywordEnabled(keyword);
        }

        /// <summary>
        /// Checks if the current end user device supports the required features in the compute shader kernel
        /// </summary>
        /// <param name="kernelIndex">The kernel index to check</param>
        /// <returns>True if the device is supported, otherwise false</returns>
        public bool IsSupported(int kernelIndex)
        {
            return computeShader.IsSupported(kernelIndex);
        }

        /// <summary>
        /// Returns the local keyword space
        /// </summary>
        /// <returns>The local keyword space</returns>
        public LocalKeywordSpace GetKeywordSpace()
        {
            return computeShader.keywordSpace;
        }

        /// <summary>
        /// Returns the local shader keywords that are currently enabled
        /// </summary>
        /// <returns>An array containing the names of the currently enabled local shader keywords</returns>
        public string[] GetShaderKeywords()
        {
            return computeShader.shaderKeywords;
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
        #endregion

        #region Disposal
        /// <summary>
        /// Disposes and removes the render texture, freeing up GPU memory
        /// </summary>
        /// <param name="name">The name of the render texture to dispose</param>
        public void DisposeRenderTexture(string name)
        {
            int index = IndexOfTextureInList(name, ref renderTextures);
            if (index != -1)
            {
                RenderTexture renderTexture = renderTextures[index];
                UpdateTextureMemoryUsage(ref renderTexture, ref localGPUMemoryUsage, MathOperation.SUBTRACT);
                renderTexture.Release();
                renderTextures.RemoveAt(index);
            }
        }

        /// <summary>
        /// Disposes and removes the global render texture, freeing up GPU memory
        /// </summary>
        /// <param name="name">The name of the global render texture to dispose</param>
        public static void DisposeGlobalRenderTexture(string name)
        {
            int index = IndexOfTextureInList(name, ref globalRenderTextures);
            if (index != -1)
            {
                RenderTexture renderTexture = globalRenderTextures[index];
                UpdateTextureMemoryUsage(ref renderTexture, ref globalGPUMemoryUsage, MathOperation.SUBTRACT);
                renderTexture.Release();
                globalRenderTextures.RemoveAt(index);
            }
        }

        /// <summary>
        /// Disposes and removes the buffer, freeing up GPU memory
        /// </summary>
        /// <param name="name">The name of the buffer to dispose</param>
        public void DisposeBuffer(string name)
        {
            int index = IndexOfBufferInList(name, ref buffers);
            if (index != -1)
            {
                UpdateBufferMemoryUsage(ref localGPUMemoryUsage, buffers[index].GetStride(), buffers[index].GetCount(), MathOperation.SUBTRACT);
                buffers[index].buffer.Dispose();
                buffers.RemoveAt(index);
            }
        }

        /// <summary>
        /// Disposes and removes the global buffer, freeing up GPU memory
        /// </summary>
        /// <param name="name">The name of the global buffer to dispose</param>
        public static void DisposeGlobalBuffer(string name)
        {
            int index = IndexOfBufferInList(name, ref globalBuffers);
            if (index != -1)
            {
                UpdateBufferMemoryUsage(ref globalGPUMemoryUsage, globalBuffers[index].GetStride(), globalBuffers[index].GetCount(), MathOperation.SUBTRACT);
                globalBuffers[index].buffer.Dispose();
                globalBuffers.RemoveAt(index);
            }
        }

        /// <summary>
        /// Disposes all buffers & render textures (local and global)
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
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }
            commandBuffer = null;

            for (int i = 0; i < renderTextures.Count; i++)
            {
                try
                {
                    if (renderTextures[i] != null)
                    {
                        renderTextures[i].Release();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError(e.Message);
                }
            }
            renderTextures.Clear();
            localGPUMemoryUsage = 0;
        }

        /// <summary>
        /// Disposes all global buffers & render textures (does not include local buffers)
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

            for (int i = 0; i < globalRenderTextures.Count; i++)
            {
                try
                {
                    if (globalRenderTextures[i] != null)
                    {
                        globalRenderTextures[i].Release();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError(e.Message);
                }
            }
            globalRenderTextures.Clear();
            globalGPUMemoryUsage = 0;
        }

        /// <summary>
        /// Disposes all local/global buffers & render textures
        /// </summary>
        public void Dispose()
        {
            DisposeLocal();
            DisposeGlobal();
        }
        #endregion
    }
}