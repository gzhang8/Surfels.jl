import CUDA
import Base: copy, deepcopy_internal, copyto!
const cu = CUDA
import ModernGL
# import QHull
# import MiniBbox



struct CamParamsC
    width::Int32
    height::Int32
    fx::Float32
    fy::Float32
    cx::Float32
    cy::Float32
    k1::Float32
    k2::Float32
    k3::Float32
    depth_factor::Float32
end

function CamParamsC(cam::CameraModel.RgbdCamParams)
    res = CamParamsC(
        Int32(cam.width),
        Int32(cam.height),
        Float32(cam.fx),
        Float32(cam.fy),
        Float32(cam.cx),
        Float32(cam.cy),
        Float32(cam.k1),
        Float32(cam.k2),
        Float32(cam.k3),
        Float32(cam.depth_factor)
    )
    return res
end


# function CameraParameters(
#     w::Int,
#     h::Int,
#     fx::FT,
#     fy::FT,
#     cx::FT,
#     cy::FT,
#     depth_factor::FT,
# ) where {FT<:AbstractFloat}
#     return CameraParameters(
#         Int32(w),
#         Int32(h),
#         Float32(fx),
#         Float32(fy),
#         Float32(cx),
#         Float32(cy),
#         0.0f0,
#         0.0f0,
#         0.0f0,
#         Float32(depth_factor),
#     )
# end
#
# augment_icl_camera() =
#     CameraParameters(640, 480, 525.0f0, 525.0f0, 319.5f0, 239.5f0, 1000.f0)
# # tum_cam() = CameraParameters()
# function CameraParameters(cam::SLAMData.CamParams, depth_factor::Float32)
#     return CameraParameters(
#         Int64(cam.w),
#         Int64(cam.h),
#         Float32(cam.fx),
#         Float32(cam.fy),
#         Float32(cam.cx),
#         Float32(cam.cy),
#         depth_factor,
#     )
# end
# struct DeepIndexMap
#     depth_map::cu.CuArray{Float32, 2}
# end
struct DeepIndexMapC
    STABLE_LAYER::Int32# =  2;
    UNSTABLE_LAYER::Int32# =  1;
    #const int REMOVE_LAYER =  1;

    # last layer of index map and depth map are for unstable vertexes
    depth_map::CUDA.CuPtr{Cfloat}
    index_map::CUDA.CuPtr{Cint}

    # remove_idx_buffer::CUDA.CuPtr{Cint}
    # remove_idx_buffer_size::Int32

    vertex_map::CUDA.CuPtr{Cfloat}
    normal_rad_map::CUDA.CuPtr{Cfloat}
    color_map::CUDA.CuPtr{Cfloat}
    confidence_map::CUDA.CuPtr{Cfloat}

    # generate sub surfel data for openGL render

    # for valid idx in index_map
    # valid_predicted_idx_d::CUDA.CuPtr{Cint}
    # predicted_surfels_count::Int32

    # generate sub surfel data for openGL render
    # predicted_surfels_data_d::CUDA.CuPtr{Cfloat}
    depth_cam::CamParamsC
end

mutable struct DeepIndexMap
    STABLE_LAYER::Int32# =  2;
    UNSTABLE_LAYER::Int32# =  1;
    #const int REMOVE_LAYER =  1;

    # last layer of index map and depth map are for unstable vertexes
    # layer x hight x width
    depth_map::cu.CuArray{Float32,3}
    index_map::cu.CuArray{Int32,3}

    # remove_idx_buffer::cu.CuArray{Int32, 1}
    # remove_idx_buffer_size_ = 1024 * 1024 * 32;// 128 MB buffer
    # remove_idx_buffer_size::Int32
    # remove_idx_count::Int32

    # 3 * layer
    vertex_map::cu.CuArray{Float32,3}
    normal_rad_map::cu.CuArray{Float32,3}
    color_map::cu.CuArray{Float32,3}
    confidence_map::cu.CuArray{Float32,3}

    # generate sub surfel data for openGL render

    # for valid idx in index_map
    # valid_predicted_idx_d::cu.CuArray{Int32, 1}
    # predicted_surfels_count::Int32

    # generate sub surfel data for openGL render
    # predicted_surfels_data_d::cu.CuArray{Float32, 1}
    depth_cam::CamParamsC
    cstruct::DeepIndexMapC
end

function DeepIndexMap(depth_cam::CameraModel.RgbdCamParams)
    height = Int64(depth_cam.height)
    width = Int64(depth_cam.width)
    pixel_num = height * width
    STABLE_LAYER, UNSTABLE_LAYER = 2, 1
    INDEX_MAP_STACK_DEPTH = STABLE_LAYER + UNSTABLE_LAYER

    depth_map = cu.CuArray{Float32,3}(undef, INDEX_MAP_STACK_DEPTH, height, width)
    index_map = cu.CuArray{Int32,3}(undef, INDEX_MAP_STACK_DEPTH, height, width)
    vertex_map = cu.CuArray{Float32,3}(undef, INDEX_MAP_STACK_DEPTH * 3, height, width)
    # normal rad
    normal_rad_map = cu.CuArray{Float32,3}(undef, INDEX_MAP_STACK_DEPTH * 3, height, width)
    # color_map
    color_map = cu.CuArray{Float32,3}(undef, INDEX_MAP_STACK_DEPTH, height, width)
    # confidence_map_
    confidence_map = cu.CuArray{Float32,3}(undef, INDEX_MAP_STACK_DEPTH, height, width)
    # valid_predicted_idx_d_
    # valid_predicted_idx_d = cu.CuArray{Int32, 1}(undef, pixel_num * INDEX_MAP_STACK_DEPTH)

    # predicted_surfels_data_d = cu.CuArray{Int32, 1}(undef, pixel_num * INDEX_MAP_STACK_DEPTH * 12)
    # predicted_surfels_count = Int32(0)

    depth_cam_c = CamParamsC(depth_cam)

    # function DeepIndexMapC(dim::DeepIndexMap)
    cstruct = DeepIndexMapC(
        STABLE_LAYER,
        UNSTABLE_LAYER,
        depth_map.storage.buffer.ptr,
        index_map.storage.buffer.ptr,
        # dim.remove_idx_buffer.buf.ptr, dim.remove_idx_buffer_size,
        vertex_map.storage.buffer.ptr,
        normal_rad_map.storage.buffer.ptr,
        color_map.storage.buffer.ptr,
        confidence_map.storage.buffer.ptr,
        # dim.valid_predicted_idx_d.buf.ptr, dim.predicted_surfels_count,
        # dim.predicted_surfels_data_d.buf.ptr,
        depth_cam_c,
    )
    # end



    return DeepIndexMap(
        Int32(STABLE_LAYER),
        Int32(UNSTABLE_LAYER),
        depth_map,
        index_map,
        # remove_idx_buffer, remove_idx_buffer_size, remove_idx_count,
        vertex_map,
        normal_rad_map,
        color_map,
        confidence_map,
        # valid_predicted_idx_d,
        # predicted_surfels_count,
        # predicted_surfels_data_d,
        depth_cam_c,
        cstruct,
    )
end

"""
set stencil values to the index map
there is a c implemtation in c_src/src/merger/deep_index_map.cu
"""
function clear!(deep_index_map::DeepIndexMap)
    deep_index_map.index_map .= -1
    deep_index_map.depth_map .= 3.402823e+38
end





# struct RemoveIndex {
# //    size_t count;
#     uint64_t max_count;
#     int* d_remove_index_ptr;
# //    int* d_remove_index_swap_ptr; // for data swap: chop rewrited indexes
#     //thrust::device_ptr<int> remove_index_th_;
#     //bool is_data_owner_;
# };

mutable struct RemoveIndex
    data::cu.CuArray{Int32}
    count::Int64
    ptr::CUDA.CuPtr{Cint}
end

"""
128 MB buffer by default
"""
function RemoveIndex(; size::Int64 = 1024 * 1024 * 32)
    data = cu.CuArray{Int32,1}(undef, size)
    return RemoveIndex(data, 0, data.storage.buffer.ptr)
end


mutable struct RgbdFrame
    time_stamp::Int32
    d_normal::CUDA.CuPtr{Cfloat}
    h_rgb::Ptr{Cuchar}
    d_rgb::CUDA.CuPtr{Cuchar}
    h_depth::Ptr{Cushort}
    # raw depth in mm
    # CudaTexture2D* raw_depth_texture
    raw_depth_texture::Ptr{Cvoid}
    # filtered depth in mm
    # CudaTexture2D* filtered_depth_texture
    filtered_depth_texture::Ptr{Cvoid}
    depth_cam::CamParamsC
    #const CameraParameters rgb_cam_;
end


# struct FrameMerger{
#     int* d_idx_buffer_raw_ptr_;//1920*1080 int size; should be enough for now
#     int* d_valid_idx_raw_ptr_;//1920*1080 int size; should be enough for now
#
#     DeepIndexMap deep_index_map_;
#     RemoveIndex remove_index_;
# };

struct FrameMergerC
    d_idx_buffer_raw_ptr::CUDA.CuPtr{Cint}#1920*1080 int size; should be enough for now
    d_valid_idx_raw_ptr::CUDA.CuPtr{Cint}#1920*1080 int size; should be enough for now

    # deep_index_map_c::DeepIndexMapC
    # remove_index_c::RemoveIndexC
end

mutable struct FrameMerger
    d_idx_buffer::cu.CuArray{Int32,1}#CUDA.CuPtr{Cint}#1920*1080 int size; should be enough for now
    d_valid_idx::cu.CuArray{Int32,1}#1920*1080 int size; should be enough for now

    # deep_index_map::DeepIndexMap
    # remove_index::cu.CuArray{Int32, 1} #1024 * 1024 * 32

    cstruct::FrameMergerC
end

function FrameMerger(cam::CameraModel.RgbdCamParams)
    d_idx_buffer = CUDA.CuArray{Int32,1}(undef, 1920 * 1080)
    d_valid_idx = CUDA.CuArray{Int32,1}(undef, 1920 * 1080)
    # remove_index = cu.CuArray{Int32, 1}(undef, 1024 * 1024 * 32)
    # remove_index_c = RemoveIndexC(UInt64(size(remove_index, 1)), remove_index.buf.ptr)

    cstruct = FrameMergerC(d_idx_buffer.storage.buffer.ptr, d_valid_idx.storage.buffer.ptr)
    return FrameMerger(d_idx_buffer, d_valid_idx, cstruct)
end


mutable struct SurfelsUtil
    time_delta::Int64
    max_depth::Float64
    confidence_thres::Float64
    weight::Float64
    cam::CamParamsC
    frame_merger::FrameMerger
    remove_index::RemoveIndex
    deep_index_map::DeepIndexMap
end


function SurfelsUtil(
    cam::CameraModel.RgbdCamParams;
    time_delta::Int64,
    max_depth::Float64,
    confidence_thres::Float64,
    weight::Float64,
)

    frame_merger = FrameMerger(cam)
    remove_index = RemoveIndex(size = 1024 * 1024 * 32)
    deep_index_map = DeepIndexMap(cam)

    return SurfelsUtil(
        time_delta,
        max_depth,
        confidence_thres,
        weight,
        CamParamsC(cam),
        frame_merger,
        remove_index,
        deep_index_map,
    )
end

# struct SurfelsData {
#     float *d_data_; // actural data storage
#     uint64_t max_vertex_count_;// max allowed vertex number based on buffer size
# };
struct SurfelsDataC
    d_data::CuPtr{Cfloat}#; // actural data storage
    max_vertex_count::UInt64#;// max allowed vertex number based on buffer size
end

mutable struct SurfelsData
    data::cu.CuArray{Float32,1}#; // actural data storage
    count::Int64#;// max allowed vertex number based on buffer size
    cstruct::SurfelsDataC
end

"""
cuda_res: cuda resource for mapping gl buffer to cuda
"""
mutable struct GlSurfelsData
    gl_vbo::ModernGL.GLuint
    cuda_res::Vector{Ptr{Cvoid}} # has to be allocated for func call
    mapped2cuda::Bool
    gl_veo::ModernGL.GLuint# index buffer
    n_facets::Int64
    bbox_corners::Matrix{Float64} # 3 x N matrix
    surfels_data::SurfelsData
end

function SurfelsData(max_count::Int64)
    surfel_data = cu.CuArray{Float32,1}(undef, max_count * 12)
    surfel_data .= -1
    surfel_data_c = SurfelsDataC(surfel_data.storage.buffer.ptr, max_count)
    return SurfelsData(surfel_data, 0, surfel_data_c)
end

function SurfelsData(cu_data::cu.CuArray{Float32,1}, count::Int64)
    # surfel_data = cu.CuArray{Float32, 1}(undef, max_count)
    max_count = Int64(size(cu_data, 1) / 12)
    surfel_data_c = SurfelsDataC(cu_data.storage.buffer.ptr, max_count)
    return SurfelsData(cu_data, count, surfel_data_c)
end

function SurfelsData(cpu_data::Array{Float32,1}, count::Int64)
    # surfel_data = cu.CuArray{Float32, 1}(undef, max_count)
    cu_data = cu.cu(cpu_data)
    max_count = Int64(size(cu_data, 1) / 12)
    surfel_data_c = SurfelsDataC(cu_data.storage.buffer.ptr, max_count)
    return SurfelsData(cu_data, count, surfel_data_c)
end

function copy(surfels_data::SurfelsData)
    cu_data::cu.CuArray{Float32,1} = copy(surfels_data.data)
    count::Int64 = surfels_data.count
    surfel_data_c = SurfelsDataC(cu_data.storage.buffer.ptr, surfels_data.cstruct.max_vertex_count)
    return SurfelsData(cu_data, count, surfel_data_c)
end


function GlSurfelsData(max_count::Int64)

    vbo = glGenBuffer()
    ModernGL.glBindBuffer(ModernGL.GL_ARRAY_BUFFER, vbo)
    ModernGL.glBufferData(
        ModernGL.GL_ARRAY_BUFFER,
        max_count * 12 * sizeof(Float32),
        C_NULL,
        ModernGL.GL_DYNAMIC_DRAW,
    )
    count = 0
    res = GlSurfelsData(vbo, count, max_count)
    res.surfels_data.data .= -1.0f0
    return res
end

function GlSurfelsData(
    cpu_data::Array{Float32,1},
    count::Int64;
    build_convex_hull::Bool = false,
    build_bbox::Bool = true,
)
    # surfel_data = cu.CuArray{Float32, 1}(undef, max_count)
    max_count = Int64(size(cpu_data, 1) / 12)
    glCheckError()
    vbo = glGenBuffer()
    @assert vbo != 0
    ModernGL.glBindBuffer(ModernGL.GL_ARRAY_BUFFER, vbo)
    ModernGL.glBufferData(
        ModernGL.GL_ARRAY_BUFFER,
        max_count * 12 * sizeof(Float32),
        cpu_data,
        ModernGL.GL_DYNAMIC_DRAW,
    )
    glCheckError()

    elementbuffer = ModernGL.GLuint(0)
    n_facets = 0
    # if build_convex_hull
    #     pts = Matrix{Float64}(undef, count, 3)

    #     for idx = 1:count
    #         pts[idx, 1] = cpu_data[(idx-1)*12+1]
    #         pts[idx, 2] = cpu_data[(idx-1)*12+2]
    #         pts[idx, 3] = cpu_data[(idx-1)*12+3]
    #     end
    #     ch = QHull.chull(pts)
    #     # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    #     # ch.points         # original points
    #     # ch.vertices       # indeices are from 0
    #     # ch.simplices      # the simplexes forming the convex hull
    #     # # show(ch)

    #     # convext_vertices = ch.points[ch.vertices.+1, :]
    #     # convext_vertices2 = pts[ch.vertices.+1, :]

    #     n_facets = size(ch.simplices, 1)
    #     indices = Vector{UInt32}(undef, n_facets * 3)

    #     for idx = 1:n_facets
    #         indices[(idx-1)*3+1:(idx-1)*3+3] .= ch.simplices[idx]
    #     end
    #     elementbuffer = glGenBuffer()
    #     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer)
    #     glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)
    #     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    # end

    bbox_corners = Matrix{Float64}(undef, 0, 0)
    if build_bbox
        # x = cpu_data[1:12:end]
        # y = cpu_data[2:12:end]
        # z = cpu_data[3:12:end]
        #
        # x = convert(Vector{Float64}, x)
        # y = convert(Vector{Float64}, y)
        # z = convert(Vector{Float64}, z)

        # rotmat, bbox_corners, volume, surface = MiniBbox.minboundbox(x, y, z)
        # TODO build mini 3D box (mini volume cube)
        bbox_corners = Matrix{Float64}(undef, 3, 8)
    end

    ModernGL.glBindBuffer(ModernGL.GL_ARRAY_BUFFER, 0)
    res = GlSurfelsData(vbo, count, max_count)
    res.gl_veo = elementbuffer
    res.n_facets = n_facets
    res.bbox_corners = bbox_corners
    return res
end

function GlSurfelsData(vbo::ModernGL.GLuint, count::Int64, max_count::Int64)

    cuda_res = [C_NULL]
    # void register_gl_buffer(GLuint gl_vertex_buffer, void**cuda_res_ptr)
    ccall(
        (:register_gl_buffer, :libsurfels),
        Cvoid,
        (ModernGL.GLuint, Ptr{Ptr{Cvoid}}),
        vbo,
        Base.unsafe_convert(Ptr{Ptr{Cvoid}}, cuda_res),
    )
    @show vbo
    @show cuda_res
    a_cuda_ptr = get_cuda_ptr_from_gl_buffer(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, cuda_res))

    # warp with CUDA
    cu_array =
        Base.unsafe_wrap(CUDA.CuArray{Float32}, a_cuda_ptr, max_count * 12, own = false)
    # cu_array .= -1
    # count = 0

    # construct data
    surfels_data = SurfelsData(cu_array, count)
    mapped2cuda = true
    veo = UInt32(0)
    n_facets = 0
    bbox_corners = Matrix{Float64}(undef, 0, 0)
    gl_surfels_data =
        GlSurfelsData(vbo, cuda_res, mapped2cuda, veo, n_facets, bbox_corners, surfels_data)
    # define finalizer
    finalizer(delete_gl_surfels_data, gl_surfels_data)
    return gl_surfels_data
end

function delete_gl_surfels_data(o::GlSurfelsData)
    # unregister gl buffer
    # void unregister_gl_cuda(void* cuda_res)
    release_cuda_ptr!(o)
    ccall((:unregister_gl_cuda, :libsurfels), Cvoid, (Ptr{Cvoid},), o.cuda_res[1])
    # converted
    if o.gl_veo != 0
        ModernGL.glDeleteBuffers(1, [o.gl_veo])
    end
    # delete vbo buffer
    ModernGL.glDeleteBuffers(1, [o.gl_vbo])
end

# float* get_cuda_ptr_from_gl_buffer(void** cuda_res_ptr)
function get_cuda_ptr_from_gl_buffer(cuda_res_ptr::Ref{Ptr{Cvoid}})
    cuda_ptr = ccall(
        (:get_cuda_ptr_from_gl_buffer, :libsurfels),
        CUDA.CuPtr{Cfloat},
        (Ptr{Ptr{Cvoid}},),
        cuda_res_ptr,
    )
    return cuda_ptr
end
function get_cuda_ptr!(o::GlSurfelsData)
    if !o.mapped2cuda
        get_cuda_ptr_from_gl_buffer(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, o.cuda_res))
        o.mapped2cuda = true
    end
end

function release_cuda_ptr!(o::GlSurfelsData)
    if o.mapped2cuda
        ccall(
            (:release_cuda_ptr_from_gl_buffer, :libsurfels),
            Cvoid,
            (Ptr{Ptr{Cvoid}},),
            Base.unsafe_convert(Ptr{Ptr{Cvoid}}, o.cuda_res),
        )
        # converted
        o.mapped2cuda = false
    end
end

# function get_cuda_ptr_from_gl_buffer()
#
# end

function copy(o::GlSurfelsData)
    # core: copy gl buffer

    # src
    # if o.mapped2cuda
    release_cuda_ptr!(o)
    #     o.mapped2cuda=false
    # end
    glCheckError()
    ModernGL.glBindBuffer(ModernGL.GL_COPY_READ_BUFFER, o.gl_vbo)
    size_vec = GLint[1]
    glCheckError()
    ModernGL.glGetBufferParameteriv(
        ModernGL.GL_COPY_READ_BUFFER,
        ModernGL.GL_BUFFER_SIZE,
        size_vec,
    )
    max_count = Int64(o.surfels_data.cstruct.max_vertex_count)
    @assert size_vec[1] == max_count * 12 * sizeof(Float32)
    glCheckError()
    # dst
    new_vbo = glGenBuffer()
    @show new_vbo
    ModernGL.glBindBuffer(ModernGL.GL_COPY_WRITE_BUFFER, new_vbo)
    ModernGL.glBufferData(
        ModernGL.GL_COPY_WRITE_BUFFER,
        max_count * 12 * sizeof(Float32),
        C_NULL,
        ModernGL.GL_DYNAMIC_DRAW,
    )
    glCheckError()
    # @show o.surfels_data.count * 12 * sizeof(Float32)
    ModernGL.glCopyBufferSubData(
        ModernGL.GL_COPY_READ_BUFFER,
        ModernGL.GL_COPY_WRITE_BUFFER,
        0,
        0,
        o.surfels_data.count * 12 * sizeof(Float32),
    )
    glCheckError()

    # copy veo
    new_veo = UInt32(0)
    if o.gl_veo != 0
        glCheckError()
        ModernGL.glBindBuffer(ModernGL.GL_COPY_READ_BUFFER, o.gl_veo)
        glCheckError()
        ModernGL.glGetBufferParameteriv(
            ModernGL.GL_COPY_READ_BUFFER,
            ModernGL.GL_BUFFER_SIZE,
            size_vec,
        )
        @assert size_vec[1] == o.n_facets * 3 * sizeof(UInt32)
        glCheckError()
        # dst
        new_veo = glGenBuffer()
        @show new_veo
        ModernGL.glBindBuffer(ModernGL.GL_COPY_WRITE_BUFFER, new_veo)
        ModernGL.glBufferData(
            ModernGL.GL_COPY_WRITE_BUFFER,
            o.n_facets * 3 * sizeof(UInt32),
            C_NULL,
            ModernGL.GL_DYNAMIC_DRAW,
        )
        glCheckError()
        ModernGL.glCopyBufferSubData(
            ModernGL.GL_COPY_READ_BUFFER,
            ModernGL.GL_COPY_WRITE_BUFFER,
            0,
            0,
            o.n_facets * 3 * sizeof(UInt32),
        )
        glCheckError()

    end
    ModernGL.glBindBuffer(ModernGL.GL_COPY_WRITE_BUFFER, 0)
    ModernGL.glBindBuffer(ModernGL.GL_COPY_READ_BUFFER, 0)
    # create
    glCheckError()
    res = GlSurfelsData(new_vbo, o.surfels_data.count, max_count)
    res.gl_veo = new_veo
    res.n_facets = o.n_facets
    res.bbox_corners = copy(o.bbox_corners)
    return res
end
function deepcopy_internal(x::Union{GlSurfelsData,SurfelsData}, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = copy(x)
    stackdict[x] = y
    return y
end

function copyto!(dst::GlSurfelsData, src::GlSurfelsData)
    glCheckError()
    @assert dst.surfels_data.cstruct.max_vertex_count >= src.surfels_data.count
    # src
    # if src.mapped2cuda
    # @show "copyto line 447"
    release_cuda_ptr!(src)
    # @show "copyto line 449"
    # src.mapped2cuda=false
    # end
    glCheckError()
    ModernGL.glBindBuffer(ModernGL.GL_COPY_READ_BUFFER, src.gl_vbo)
    glCheckError()
    # dst
    # if dst.mapped2cuda
    # @show "copyto line 455"
    release_cuda_ptr!(dst)
    # @show "copyto line 457"
    # dst.mapped2cuda=false
    # end
    glCheckError()
    ModernGL.glBindBuffer(ModernGL.GL_COPY_WRITE_BUFFER, dst.gl_vbo)
    glCheckError()
    size_vec = GLint[1]
    glCheckError()
    ModernGL.glGetBufferParameteriv(
        ModernGL.GL_COPY_READ_BUFFER,
        ModernGL.GL_BUFFER_SIZE,
        size_vec,
    )
    # if
    data_count = src.surfels_data.count * 12 * sizeof(Float32)
    @assert size_vec[1] >= data_count "vbo:$(src.gl_vbo) buf size: $(size_vec[1]) >= data_count $data_count"
    # @show "src_size: $(size_vec[1])"
    ModernGL.glGetBufferParameteriv(
        ModernGL.GL_COPY_WRITE_BUFFER,
        ModernGL.GL_BUFFER_SIZE,
        size_vec,
    )
    # @show "dst_size: $(size_vec[1])"
    @assert size_vec[1] >= data_count "vbo:$(dst.gl_vbo) buf size: $(size_vec[1]) >= data_count $data_count"
    # copy data
    # @show src.surfels_data.count * 12 * sizeof(Float32)
    ModernGL.glCopyBufferSubData(
        ModernGL.GL_COPY_READ_BUFFER,
        ModernGL.GL_COPY_WRITE_BUFFER,
        0,
        0,
        data_count,
    )
    glCheckError()
    ModernGL.glBindBuffer(ModernGL.GL_COPY_WRITE_BUFFER, 0)
    glCheckError()
    ModernGL.glBindBuffer(ModernGL.GL_COPY_READ_BUFFER, 0)
    glCheckError()

    if length(src.bbox_corners) == 24
        if length(dst.bbox_corners) == 24
            copyto!(dst.bbox_corners, src.bbox_corners)
        else
            dst.bbox_corners = copy(src.bbox_corners)
        end
    end
end

function copytogl(o::SurfelsData)
    max_count = Int64(o.surfels_data.cstruct.max_vertex_count)
    gl_data = GlSurfelsData(max_count)
    @assert gl_data.mapped2cuda
    copyto!(gl_data.surfels_data.data, o.data)
    gl_data.surfels_data.count = o.count
    return gl_data
end


function bbox_in_view(
    o::GlSurfelsData,
    Tcw::Matrix{Float64},
    cam::CameraModel.RgbdCamParams)
    # project all points
    # [x, y, z]
    # x = Vector{Float64}(undef, 8)
    # y = Vector{Float64}(undef, 8)
    # z = Vector{Float64}(undef, 8)
    corners_w = o.bbox_corners
    @assert size(corners_w, 1) == 3
    corners_c = CameraModel.applyT(Tcw, corners_w)

    z = corners_c[3, :]
    x = cam.fx ./ z .* corners_c[1, :] .+ cam.cx
    y = cam.fy ./ z .* corners_c[2, :] .+ cam.cy

    neg_z_count = 0
    left_count = 0
    right_count = 0
    top_count = 0
    down_count = 0
    for i = 1:8
        if z[i] < 0.0
            neg_z_count += 0
        end

        if x[i] < 0
            left_count += 1
        elseif x[i] > cam.width
            right_count += 1
        end

        if y[i] < 0
            top_count += 1
        elseif y[i] > cam.height
            down_count += 1
        end
    end
    if (
        (neg_z_count == 8) |
        (left_count == 8) |
        (right_count == 8) |
        (down_count == 8) |
        (top_count == 8)
    )
        return false
    else
        return true
    end
end


# see https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/index.html#Struct-Type-correspondences-1
struct Mat4C
    # // initialize a Mat3 to identity by default
    data::NTuple{16,Cfloat}#[16];#//{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
end


function Mat4C(mat4::Matrix{Float64})
    vec = convert(Vector{Float32}, mat4'[:])
    # @show vec
    return Mat4C(Tuple(vec))
end
