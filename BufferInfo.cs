using System;
using UnityEngine;

namespace GPUComputeModule
{
    /// <summary>
    /// Contains the ComputeBuffer, additional information about the buffer, and helper functions for retreiving/setting non-data specific buffer values
    /// </summary>
    public struct BufferInfo
    {
        public string name;
        public ComputeBuffer buffer;
        public Type dataType;
        public ComputeBufferType computeBufferType;
        public ComputeBufferMode computeBufferMode;
        public int[] kernelIndices;

        public BufferInfo(string name, ComputeBuffer buffer, Type dataType, ComputeBufferType computeBufferType, ComputeBufferMode computeBufferMode, int[] kernelIndices)
        {
            this.name = name;
            this.buffer = buffer;
            this.dataType = dataType;
            this.computeBufferType = computeBufferType;
            this.computeBufferMode = computeBufferMode;
            this.kernelIndices = kernelIndices;
        }

        /// <summary>
        /// Returns the size of one element in the buffer (in bytes)
        /// </summary>
        /// <returns>The size of one element in the buffer</returns>
        public int GetStride()
        {
            return buffer.stride;
        }

        /// <summary>
        /// Gets the number of elements in the buffer (note: for append/counter type buffers this returns the initial capacity)
        /// </summary>
        /// <returns>The number of elements in the buffer (for append/counter type buffers, its total capacity)</returns>
        public int GetCount()
        {
            return buffer.count;
        }

        /// <summary>
        /// Gets the number of elements in the append/counter type buffer by writing the value from the GPU to the CPU via a count buffer
        /// </summary>
        /// <returns>The number of elements in the append/counter type buffer</returns>
        public int GetAppendCountFromGPU()
        {
            if (computeBufferType != ComputeBufferType.Append && computeBufferType != ComputeBufferType.Counter)
            {
                Debug.LogWarning("The buffer isn't an append/counter type buffer");
                return -1;
            }
            uint[] appendCount = new uint[1];
            ComputeBuffer countBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Raw);
            ComputeBuffer.CopyCount(buffer, countBuffer, 0);
            countBuffer.GetData(appendCount);
            countBuffer.Dispose();
            return (int)appendCount[0];
        }

        /// <summary>
        /// Sets the counter value of an append/counter type buffer
        /// </summary>
        /// <param name="counterValue">The value to set the counter to</param>
        public void SetCounterValue(uint counterValue)
        {
            if (computeBufferType != ComputeBufferType.Append && computeBufferType != ComputeBufferType.Counter)
            {
                Debug.LogWarning("The buffer isn't an append/counter type buffer");
                return;
            }
            buffer.SetCounterValue(counterValue);
        }

        /// <summary>
        /// Returns the number of bytes this buffer is using in GPU memory
        /// </summary>
        /// <returns>The number of bytes used by the buffer</returns>
        public int GetByteCount()
        {
            return GetCount() * GetStride();
        }

        /// <summary>
        /// Converts the input byte count to its relevant unit (e.g. 1000 bytes = "1 KB")
        /// </summary>
        /// <returns>A formatted string containing the converted value followed by its unit</returns>
        public string GetByteCountAsFormattedString()
        {
            return GPUCompute.ByteCountToFormattedString(GetByteCount());
        }

        /// <summary>
        /// Returns a string containing all buffer information, and all values inside the buffer for debugging purposes
        /// This copies all values from the GPU to the CPU, and should only be used for debugging
        /// </summary>
        /// <returns>A string containing all buffer information, and all values inside the buffer</returns>
        public string GetDebugString()
        {
            string kernels = "";
            for (int i = 0; i < kernelIndices.Length - 1; i++)
            {
                kernels += kernelIndices[i] + ", ";
            }

            if (kernelIndices.Length > 0)
            {
                kernels += kernelIndices[kernelIndices.Length - 1];
            }

            string length = "Length: [" + GetCount().ToString() + "]";
            if (computeBufferType == ComputeBufferType.Append || computeBufferType == ComputeBufferType.Counter)
            {
                length = "Count: [" + GetAppendCountFromGPU().ToString() + "], " + length;
            }

            return "Buffer Name: [" + name +
            "], Data Type: [" + dataType.ToString() +
            "], " + length +
            ", Stride: [" + GetStride() +
            "], VRAM Usage: [" + GetByteCountAsFormattedString() +
            "], Buffer Type: [" + computeBufferType.ToString() +
            "], Buffer Mode: [" + computeBufferMode.ToString() +
            "], Kernel Indices: [" + kernels +
            "]";
        }
    }
}