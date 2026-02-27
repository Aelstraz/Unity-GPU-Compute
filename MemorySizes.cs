using System;
using System.Runtime.InteropServices;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace UsefulUtilities
{
    public static class MemorySizes
    {
        public static int GetStructSizeInBytes<T>() where T : struct
        {
            int stride = -1;
            try
            {
                stride = Marshal.SizeOf<T>();
            }
            catch (Exception e)
            {
                Debug.LogError("Unable to get stride value of struct: " + typeof(T).Name + ", " + e.Message);
            }
            return stride;
        }

        public static int GetStructSizeInBytes(Type type)
        {
            int stride = -1;
            if (!type.IsStruct())
            {
                Debug.LogError("Unable to get stride value of Type: " + type.Name + ", type is not a struct");
            }
            try
            {
                stride = Marshal.SizeOf(type);
            }
            catch (Exception e)
            {
                Debug.LogError("Unable to get stride value of Type: " + type.Name + ", " + e.Message);
            }
            return stride;
        }

        /// <param name="renderTextureDepthBits">The precision of the render textures depth buffer in bits (only used if the RenderTextureFormat is set to Depth)</param>
        public static int GetBitsPerPixel(RenderTextureFormat renderTextureFormat, int renderTextureDepthBits = 16)
        {
#pragma warning disable CS0618 // Type or member is obsolete
            switch (renderTextureFormat)
            {
                case RenderTextureFormat.Shadowmap:
                case RenderTextureFormat.Depth:
                    return renderTextureDepthBits;
                case RenderTextureFormat.R8:
                    return 8;
                case RenderTextureFormat.ARGB1555:
                case RenderTextureFormat.ARGB4444:
                case RenderTextureFormat.R16:
                case RenderTextureFormat.RG16:
                case RenderTextureFormat.RGB565:
                case RenderTextureFormat.RHalf:
                    return 16;
                case RenderTextureFormat.BGR101010_XR:
                    return 30;
                case RenderTextureFormat.ARGB2101010:
                case RenderTextureFormat.ARGB32:
                case RenderTextureFormat.BGRA32:
                case RenderTextureFormat.RFloat:
                case RenderTextureFormat.RG32:
                case RenderTextureFormat.RGB111110Float:
                case RenderTextureFormat.RGHalf:
                case RenderTextureFormat.RInt:
                    return 32;
                case RenderTextureFormat.BGRA10101010_XR:
                    return 40;
                case RenderTextureFormat.ARGB64:
                case RenderTextureFormat.ARGBHalf:
                case RenderTextureFormat.RGBAUShort:
                case RenderTextureFormat.RGFloat:
                case RenderTextureFormat.RGInt:
                    return 64;
                case RenderTextureFormat.ARGBFloat:
                case RenderTextureFormat.ARGBInt:
                    return 128;
            }
#pragma warning restore CS0618 // Type or member is obsolete
            return 0;
        }

        public static int GetBitsPerPixel(TextureFormat format)
        {
#pragma warning disable CS0618 // Type or member is obsolete
            switch (format)
            {
                case TextureFormat.EAC_R:
                case TextureFormat.EAC_R_SIGNED:
                case TextureFormat.BC4:
                case TextureFormat.DXT1:
                case TextureFormat.DXT1Crunched:
                case TextureFormat.ETC2_RGB:
                case TextureFormat.ETC_RGB4:
                case TextureFormat.ETC_RGB4Crunched:
                case TextureFormat.ETC2_RGBA1:
                    return 4;
                case TextureFormat.Alpha8:
                case TextureFormat.R8:
                case TextureFormat.R8_SIGNED:
                case TextureFormat.BC5:
                case TextureFormat.BC6H:
                case TextureFormat.BC7:
                case TextureFormat.DXT5:
                case TextureFormat.DXT5Crunched:
                case TextureFormat.ETC2_RGBA8:
                case TextureFormat.ETC2_RGBA8Crunched:
                case TextureFormat.EAC_RG:
                case TextureFormat.EAC_RG_SIGNED:
                    return 8;
                case TextureFormat.ARGB4444:
                case TextureFormat.R16:
                case TextureFormat.R16_SIGNED:
                case TextureFormat.RG16:
                case TextureFormat.RG16_SIGNED:
                case TextureFormat.RGB565:
                case TextureFormat.RGBA4444:
                case TextureFormat.YUY2:
                    return 16;
                case TextureFormat.RGB24:
                case TextureFormat.RGB24_SIGNED:
                    return 24;
                case TextureFormat.RGB9e5Float:
                    return 27;
                case TextureFormat.ARGB32:
                case TextureFormat.BGRA32:
                case TextureFormat.RFloat:
                case TextureFormat.RG32:
                case TextureFormat.RG32_SIGNED:
                case TextureFormat.RGBA32:
                case TextureFormat.RGBA32_SIGNED:
                case TextureFormat.RGHalf:
                    return 32;
                case TextureFormat.RGB48:
                case TextureFormat.RGB48_SIGNED:
                    return 48;
                case TextureFormat.RGBA64:
                case TextureFormat.RGBA64_SIGNED:
                case TextureFormat.RGBAHalf:
                case TextureFormat.RGFloat:
                    return 64;
                case TextureFormat.ASTC_10x10:
                case TextureFormat.ASTC_12x12:
                case TextureFormat.ASTC_4x4:
                case TextureFormat.ASTC_5x5:
                case TextureFormat.ASTC_6x6:
                case TextureFormat.ASTC_8x8:
                case TextureFormat.ASTC_HDR_10x10:
                case TextureFormat.ASTC_HDR_12x12:
                case TextureFormat.ASTC_HDR_4x4:
                case TextureFormat.ASTC_HDR_5x5:
                case TextureFormat.ASTC_HDR_6x6:
                case TextureFormat.ASTC_HDR_8x8:
                case TextureFormat.RGBAFloat:
                    return 128;
            }
#pragma warning restore CS0618 // Type or member is obsolete
            return 0;
        }

        public static int GetBitsPerPixel(GraphicsFormat format)
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
    }
}
