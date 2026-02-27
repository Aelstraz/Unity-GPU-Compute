# GPU Compute (Unity)
GPU Compute provides the ultimate & easiest way to setup & execute GPU compute shaders in Unity (additional features may be undocumented here).

- Reduces the amount of code and complexity to execute a compute shader
- Create, edit and read buffers easily (buffer strides & lengths are calculated automatically)
- Automatically calculate optimal GPU thread group sizes for your workload
- Asynchronous execution
- Track GPU memory usage
- Track compute execution time
- Buffer debugging
- Centralized in a single class
- Plus more!

![alt text](https://github.com/Aelstraz/Unity-GPU-Compute/blob/main/Screenshot.png?raw=true)

### Importing/Installing:
Simply add the entire folder into your projects Assets folder.
***
### Instantiate:
Start by creating a new instance of GPU Compute, providing the compute shader you want to use:

	GPUCompute gpuCompute = new GPUCompute(myComputeShader);

***
### Dispose:
When you are done with your GPU Compute instance, make sure to dispose it to prevent memory leaks on the GPU- using one of the disposal methods as followed:

	gpuCompute.Dipose();
 	gpuCompute.DisposeLocal();
  	gpuCompute.DisposeGlobal();

***
### Setting & Getting Buffers/Textures/Variables:
To create a buffer simply input the buffer name, and the actual data to be passed into the buffer (the data struct type and length of the data is used to automatically set the size of the buffer). Empty buffers can also be made by supplying the data struct type and length:
	
 	gpuCompute.SetBuffer("vertices", ref vertices);
 	gpuCompute.SetBuffer("uvs", ref uvs);
  	gpuCompute.CreateEmptyBuffer<int>("myEmptyBuffer", myEmptyBufferLength);
 	
Buffer data, textures and variables values can be set like so:

	gpuCompute.SetInt("myInt", myInt);
 	gpuCompute.SetFloat("myFloat", myFloat);
  	gpuCompute.SetVector("myVector", myVector);
   	gpuCompute.SetBuffer("myVectorBuffer", ref myVectorBuffer);
   	gpuCompute.SetRenderTexture("myRenderTexture", myRenderTexture);

Buffer data can be retrieved as shown (a reference list/array of the same data type and length must be supplied to write the buffer data to):

	Vector3[] myVectorBuffer = new Vector3[100];
	gpuCompute.GetBufferData("myVectorBuffer", ref myVectorBuffer);

***
### Global Buffers/Textures:
Global buffers can be set and retrieved in the same way as local buffers:

	GPUCompute.SetGlobalBuffer("globalVertices", ref vertices);	
	GPUCompute.CreateEmptyGlobalBuffer<Vector3>("myEmptyGlobalBuffer", myEmptyGlobalBufferLength);
 	GPUCompute.SetBuffer("myGlobalVectorBuffer", ref myGlobalVectorBuffer);
  	GPUCompute.GetBufferData("myGlobalUVBuffer", ref myGlobalUVBuffer);
 
In order to use them in your compute shader, they first need to be linked to your GPU Compute instance as shown:

	gpuCompute.LinkGlobalBuffer("globalBufferName");
	gpuCompute.LinkGlobalTexture("globalTextureName", "globalTextureName");
 
***
### Setting Thread Group Size:
Thread group sizes can be automatically calculated based on your workload dimensions. Select one of the dimension options below (1D, 2D, 3D) based on your workload needs. Thread group sizes can also be set manually.
***
### Thread Group Size (1D):
For a one-dimensional workload (e.g. an array) pass your job length and shader number of threads into the SetCalculatedThreadGroupSize() method:

	int jobLength = vertices.Length;
	gpuCompute.SetCalculatedThreadGroupSize(jobLength, shaderNumOfThreads);

You will likely need to work out the index of the thread in your compute shader, so the SetCalculatedThreadGroupSize() method also returns the dimension width. The dimension width can be passed into your compute shader in order to get the current index, as seen below:

	int threadWidth = gpuCompute.SetCalculatedThreadGroupSize(jobLength, shaderNumOfThreads);
	gpuCompute.SetInt("threadWidth", threadWidth);

Then in your compute shader the current index can be found like so:

	void CSMain(uint3 id : SV_DispatchThreadID)
	{
		uint index = id.x + id.y * threadWidth;
	}
 
***
### Thread Group Size (2D):
For a two-dimensional workload (e.g. a texture) pass the width, length and your compute shader number of threads into the SetCalculatedThreadGroupSize2D() method:

	gpuCompute.SetCalculatedThreadGroupSize2D(width, length, shaderNumOfThreads);

***
### Setting Thread Group Size (3D):
For a three-dimensional workload (e.g. a volume) pass the width, length, depth and your compute shader number of threads into the SetCalculatedThreadGroupSize3D() method:

	gpuCompute.SetCalculatedThreadGroupSize3D(width, length, depth, shaderNumOfThreads);

***
### Executing:
After setting any required buffers/variables, and the thread group size, the compute shader can be executed. 
For standard execution (non async) simply run the following:

 	gpuCompute.Execute();

For async execution first subscribe to the OnExecuteComplete event as shown below:

	gpuCompute.OnExecuteComplete += GPUCompute_OnExecuteComplete;
 
Then run the ExecuteAsync() method as a coroutine:
	
	StartCoroutine(gpuCompute.ExecuteAsync());
  
***
### Get GPU Memory Used:
Memory used by the GPU is automatically tracked and can be viewed as either total bytes, or a formatted string.

 	gpuCompute.GetGPUMemoryUsed();
	gpuCompute.GetGPUMemoryUsedFormatted();

 	gpuCompute.GetGlobalGPUMemoryUsed();
	gpuCompute.GetGlobalGPUMemoryUsedFormatted();

 	gpuCompute.GetTotalGPUMemoryUsed();
	gpuCompute.GetTotalGPUMemoryUsedFormatted();
 
 ***
### Get Compute Time:
The amount of time it took for the last computation to finish is also tracked and viewable through:

 	TimeSpan lastComputeTime = gpuCompute.GetLastComputeTime();
  
***
### Get BufferInfo/ComputeBuffer:
BufferInfo containing the buffer data type, name and the associated ComputeBuffer can be retreived through the following methods (depending if the buffer is local or global):

 	BufferInfo bufferInfo = gpuCompute.GetBufferInfo(name);
	BufferInfo globalBufferInfo = gpuCompute.GetGlobalBufferInfo(name);
  
***
### Twitter: [@aelstraz](https://x.com/Aelstraz)
