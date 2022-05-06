import FileIO: save
using Images
import ModernGL

function delete_curgbd!(rgbdframe::RgbdFrame)
    ccall((:delete_rgbd_frame, :libsurfels), Cvoid, (RgbdFrame,), rgbdframe)

end

function preprocess_rgbd(
    image::Matrix{RGB{Normed{UInt8,8}}},
    depth::Matrix{UInt16},
    cam::CamParamsC,
    timestamp::Int,
)
    rgbdframe = ccall(
        (:create_rgbd_frame, :libsurfels),
        RgbdFrame,
        (Ptr{Cuchar}, CamParamsC, Ptr{Cushort}, CamParamsC, Cint),
        Base.unsafe_convert(Ptr{Cuchar}, image),
        cam,
        Base.unsafe_convert(Ptr{Cushort}, depth),
        cam,
        Int32(timestamp),
    )
    ccall((:preProcessing, :libsurfels), Cvoid, (RgbdFrame,), rgbdframe)
    finalizer(delete_curgbd!, rgbdframe)
    return rgbdframe
end




function merge_first_frame!(
    frame_merger::FrameMerger,
    surfels_data::SurfelsData,
    # rgbd_frame::SLAMData.RGBDFrame,
    image::Matrix{RGB{Normed{UInt8,8}}},
    depth::Matrix{UInt16},
    pose::Matrix{Float64},
    timestamp::Int64,
    max_depth::Float64,
    cam::CamParamsC,
    weight::Float64;
    culabel::cu.CuArray{Int32,2} = CUDA.CuArray{Int32,2}(undef, 0, 0),
)

    rgbdframe = ccall(
        (:create_rgbd_frame, :libsurfels),
        RgbdFrame,
        (Ptr{Cuchar}, CamParamsC, Ptr{Cushort}, CamParamsC, Cint),
        Base.unsafe_convert(Ptr{Cuchar}, image),
        cam,
        Base.unsafe_convert(Ptr{Cushort}, depth),
        cam,
        Int32(timestamp),
    )

    # void preProcessing(RgbdFrame rgbd_frame);
    ccall((:preProcessing, :libsurfels), Cvoid, (RgbdFrame,), rgbdframe)




    valid_count = ccall(
        (:valid_depth, :libsurfels),
        Clonglong,
        (FrameMergerC, RgbdFrame, CamParamsC),
        frame_merger.cstruct,
        rgbdframe,
        cam,
    )

    # mat4_jl = Matrix{Float32}(LinearAlgebra.I, 4, 4)
    ## --- pass a pose matrix as cpu heap arrays
    mat4_c = Mat4C(pose)

    d_label_ptr = CUDA.CU_NULL
    if length(culabel) != 0
        d_label_ptr = culabel.storage.buffer.ptr
    end

    surfels_data.count = ccall(
        (:add_new_surfels, :libsurfels),
        Clonglong,
        (
            FrameMergerC,
            SurfelsDataC,
            Clonglong,
            RgbdFrame,
            CUDA.CuPtr{Cint},
            Mat4C,
            CamParamsC,
            Cint,
            Cfloat,
            Clonglong,
        ),
        frame_merger.cstruct,
        surfels_data.cstruct,
        surfels_data.count,
        rgbdframe,
        d_label_ptr,
        mat4_c,
        cam,
        Int32(timestamp),
        Float32(weight),
        valid_count,
    )
    # delete data in rgbd frame
    ccall((:delete_rgbd_frame, :libsurfels), Cvoid, (RgbdFrame,), rgbdframe)
end


function merge_addtional_frame!(
    frame_merger::FrameMerger,
    surfel_data::SurfelsData,
    deep_index_map::DeepIndexMap,
    remove_index::RemoveIndex,
    image::Matrix{RGB{Normed{UInt8,8}}},
    depth::Matrix{UInt16},
    pose::Matrix{Float64},
    timestamp::Int64,
    timedelta::Int64,
    max_depth::Float64,
    confidence_thres::Float64,
    cam::CamParamsC,
    weight::Float64;
    culabel::cu.CuArray{Int32,2} = CUDA.CuArray{Int32,2}(undef, 0, 0),
    culabel_predict::cu.CuArray{Int32,2} = CUDA.CuArray{Int32,2}(undef, 0, 0),
    remove_old::Bool = false,
)

    mat4_c = Mat4C(pose)
    # mat4_inv_c = Mat4C(inv(pose))

    # depth_view = Gray.(adframe.depth/4000);
    rgbdframe = preprocess_rgbd(image, depth, cam, timestamp)

    d_label_ptr = CUDA.CU_NULL
    if length(culabel) != 0
        d_label_ptr = culabel.storage.buffer.ptr
    end

    d_label_predict_ptr = CUDA.CU_NULL
    if length(culabel_predict) != 0
        d_label_predict_ptr = culabel_predict.storage.buffer.ptr
    end

    # get index map
    ccall(
        (:predict_index_map, :libsurfels),
        Cvoid,
        (
            SurfelsDataC,
            Clonglong,
            DeepIndexMapC,
            CUDA.CuPtr{Cint},
            Mat4C,
            CamParamsC,
            Cfloat,
            Cfloat,
            Cint,
            Cint,
        ),
        surfel_data.cstruct,
        surfel_data.count,
        deep_index_map.cstruct,
        d_label_predict_ptr,
        mat4_c,
        cam,
        Float32(max_depth),
        Float32(confidence_thres),
        Int32(timestamp),
        Int32(timedelta),
    )

    # aidx = deep_index_map.index_map[:, 389+1, 453+1]
    # ddd = deep_index_map.depth_map[:, 389+1, 453+1]

    # adepthlayer = collect(deep_index_map.depth_map[3, :, :])
    # adepthlayer[adepthlayer.> 5] .= 0.0;
    # adepthlayer_img = Gray.(adepthlayer ./ 4)

    # Mark remove indexes

    # // remove_idx_buffer has to be longer than surfel_count
    # uint64_t FlagRemoveSuefels(SurfelsData global_model,
    #                            int64_t surfel_count,
    #                            float confidence_thres,
    #                            int time_stamp,
    #                            int time_delta,
    #                            int* remove_idx_buffer,
    #                            int remove_idx_buffer_size)

    remove_index.count = ccall(
        (:FlagRemoveSuefels, :libsurfels),
        Clonglong,
        (SurfelsDataC, Clonglong, Cfloat, Cint, Cint, CUDA.CuPtr{Cint}, Cint, Cuchar),
        surfel_data.cstruct,
        surfel_data.count,
        Float32(confidence_thres),
        Int32(timestamp),
        Int32(timedelta),
        remove_index.ptr,
        Int32(size(remove_index.data, 1)),
        remove_old,
    )


    # SurfelsAssociation

    # void data_assosiation( FrameMerger merger,
    # RgbdFrame frame,
    # DeepIndexMap deep_index_map_,
    # CamParamsC depth_cam,
    # float max_depth)
    ccall(
        (:data_assosiation, :libsurfels),
        Cvoid,
        (FrameMergerC, RgbdFrame, DeepIndexMapC, CamParamsC, Cfloat),
        frame_merger.cstruct,
        rgbdframe,
        deep_index_map.cstruct,
        cam,
        Float32(max_depth),
    )

    # if update_count > 0: then cudaUpdateSurfels
    # int64_t prepare_update_mask(FrameMerger merger,  CamParamsC depth_cam){

    merge_count = ccall(
        (:prepare_update_mask, :libsurfels),
        Clonglong,
        (FrameMergerC, CamParamsC),
        frame_merger.cstruct,
        cam,
    )
    if merge_count > 0
        # void update_surfels(FrameMerger merger,
        #                 SurfelsData model,
        #                 int64_t surfel_count,
        #                 RgbdFrame frame,
        #                 Mat4 pose_mat4,
        #                 float weight,
        #                 float max_depth,
        #                 int update_count)
        ccall(
            (:update_surfels, :libsurfels),
            Cvoid,
            (
                FrameMergerC,
                SurfelsDataC,
                Clonglong,
                RgbdFrame,
                CUDA.CuPtr{Cint},
                Mat4C,
                Cfloat,
                Cfloat,
                Cint,
            ),
            frame_merger.cstruct,
            surfel_data.cstruct,
            surfel_data.count,
            rgbdframe,
            d_label_ptr,
            mat4_c,
            Float32(weight),
            Float32(max_depth),
            Int32(merge_count),
        )
    end
    # add new surfel if no enough old surfels to remove
    add_count = ccall(
        (:prepare_add_mask, :libsurfels),
        Clonglong,
        (FrameMergerC, CamParamsC),
        frame_merger.cstruct,
        cam,
    )
    if remove_index.count < add_count

        # check if adding more surfels would go beyond capacity
        if add_count + surfel_data.count < surfel_data.cstruct.max_vertex_count

            surfel_data.count = ccall(
                (:add_new_surfels, :libsurfels),
                Clonglong,
                (
                    FrameMergerC,
                    SurfelsDataC,
                    Clonglong,
                    RgbdFrame,
                    CUDA.CuPtr{Cint},
                    Mat4C,
                    CamParamsC,
                    Cint,
                    Cfloat,
                    Clonglong,
                ),
                frame_merger.cstruct,
                surfel_data.cstruct,
                surfel_data.count,
                rgbdframe,
                d_label_ptr,
                mat4_c,
                cam,
                Int32(timestamp),
                Float32(weight),
                add_count,
            )
        else
            throw("surfel_data reached its max capacity, new surfels are skipped!")
        end
    else
        #       __host__ void overwrite_surfels(FrameMerger merger,
        #                                 SurfelsData model,
        #                                 RgbdFrame frame,
        #                                 Mat4 pose_mat4,
        #                                 CamParamsC depth_cam,
        #                                 float weight,
        #                                 int* d_remove_index_buffer,
        #                                 int add_count){
        ccall(
            (:overwrite_surfels, :libsurfels),
            Cvoid,
            (
                FrameMergerC,
                SurfelsDataC,
                RgbdFrame,
                CUDA.CuPtr{Cint},
                Mat4C,
                CamParamsC,
                Cfloat,
                CUDA.CuPtr{Cint},
                Cint,
            ),
            frame_merger.cstruct,
            surfel_data.cstruct,
            rgbdframe,
            d_label_ptr,
            mat4_c,
            cam,
            Float32(weight),
            remove_index.ptr,
            Int32(add_count),
        )
    end
    # get index map

    # SurfelsAssociation

    # if update_count > 0: then cudaUpdateSurfels

    # add new surfel if no enough old surfels to remove

    # overwrite if old is enough to remove
    return surfel_data.count

end

# function merge_frame!(frame_merger::FrameMerger,
#                       surfel_data::SurfelsData,
#                       deep_index_map::DeepIndexMap,
#                       remove_index::RemoveIndex,
#                       # rgbd_frame_cpu::SLAMData.RGBDFrame,
#                       image::Matrix{RGB{Normed{UInt8,8}}},
#                       depth::Matrix{UInt16},
#                       pose::Matrix{Float64},
#                       timestamp::Int64,
#                       timedelta::Int64,
#                       max_depth::Float64,
#                       confidence_thres::Float64,
#                       cam::CamParamsC,
#                       weight::Float64;
#                       culabel::cu.CuArray{Int32, 2}=CUDA.CuArray{Int32, 2}(undef, 0, 0),
#                       culabel_predict::cu.CuArray{Int32, 2}=CUDA.CuArray{Int32, 2}(undef, 0, 0),
#                       remove_old::Bool= false)

"""
note: change default parameters will just change the values for once;
      if you want to change it permernently, change the ones in surfels_util

      remove_old: it will happen before that merging, this will remove all the
                  old (current_t - t < timedelta) surfels, not matter confident
                  or not.
                  If you want to clear all the points, set surfel_data.count to
                  0 instead.
"""
function merge_frame!(
    surfels_util::SurfelsUtil,
    surfel_data::SurfelsData,
    image::Matrix{RGB{Normed{UInt8,8}}},
    depth::Matrix{UInt16},
    pose::Matrix{Float64},
    timestamp::Int64;
    timedelta::Int64 = surfels_util.time_delta,
    max_depth::Float64 = surfels_util.max_depth,
    confidence_thres::Float64 = surfels_util.confidence_thres,
    cam::CamParamsC = surfels_util.cam,
    weight::Float64 = surfels_util.weight,
    culabel::cu.CuArray{Int32,2} = CUDA.CuArray{Int32,2}(undef, 0, 0),
    culabel_predict::cu.CuArray{Int32,2} = CUDA.CuArray{Int32,2}(undef, 0, 0),
    remove_old::Bool = false,
)

    if surfel_data.count == 0
        # @show surfel_data.count
        merge_first_frame!(
            surfels_util.frame_merger,
            surfel_data,
            image,
            depth,
            pose,
            timestamp,
            max_depth,
            cam,
            weight;
            culabel = culabel,
        )
    else
        # @show surfel_data.count
        merge_addtional_frame!(
            surfels_util.frame_merger,
            surfel_data,
            surfels_util.deep_index_map,
            surfels_util.remove_index,
            # rgbd_frame_cpu::SLAMData.RGBDFrame,
            image,
            depth,
            pose,
            timestamp,
            timedelta,
            max_depth,
            confidence_thres,
            cam,
            weight;
            culabel = culabel,
            culabel_predict = culabel_predict,
            remove_old = remove_old,
        )
    end
end



function predict_label!(#frame_merger::FrameMerger,
    surfel_data::SurfelsData,
    deep_index_map::DeepIndexMap,
    pose::Matrix{Float64},
    timestamp::Int64,
    timedelta::Int64,
    max_depth::Float64,
    confidence_thres::Float64,
    cam::CamParamsC,
    weight::Float64,
    culabel_predict::cu.CuArray{Int32,2},
)
    culabel_predict .= 0
    mat4_c = Mat4C(pose)
    # mat4_inv_c = Mat4C(inv(pose))
    # get index map
    ccall(
        (:predict_index_map, :libsurfels),
        Cvoid,
        (
            SurfelsDataC,
            Clonglong,
            DeepIndexMapC,
            CUDA.CuPtr{Cint},
            Mat4C,
            CamParamsC,
            Cfloat,
            Cfloat,
            Cint,
            Cint,
        ),
        surfel_data.cstruct,
        surfel_data.count,
        deep_index_map.cstruct,
        culabel_predict.storage.buffer.ptr,
        mat4_c,
        cam,
        Float32(max_depth),
        Float32(confidence_thres),
        Int32(timestamp),
        Int32(timedelta),
    )

end


function applyT!(T::Matrix{Float64}, surfel::SurfelsData)
    # void applyTSurfels(Mat4 T, SurfelsData model, int64_t model_count);
    mat4_c = Mat4C(T)
    ccall(
        (:applyTSurfels, :libsurfels),
        Cvoid,
        (Mat4C, SurfelsDataC, Clonglong),
        mat4_c,
        surfel.cstruct,
        surfel.count,
    )
end

function applyT!(T::Matrix{Float64}, surfel::GlSurfelsData)
    # void applyTSurfels(Mat4 T, SurfelsData model, int64_t model_count);
    get_cuda_ptr!(surfel)
    applyT!(T, surfel.surfels_data)
    if length(surfel.bbox_corners) > 0
        CameraModel.applyT!(T, surfel.bbox_corners)
    end
end

function observe(
    surfel_data_collections::Union{
        Base.ValueIterator{Dict{Int64,SurfelsData}},
        Vector{SurfelsData},
    },
    Twc::Matrix{Float64},
    cam::CamParamsC,
    max_depth::Float64,
    confidence_thres::Float64,
    timestamp::Int64,
    timedelta::Int64;
    deep_index_map = DeepIndexMap(cam),
)

    d_label_predict_ptr = CUDA.CU_NULL
    # if length(culabel_predict) != 0
    #       d_label_predict_ptr = culabel_predict.buf.ptr
    # end
    Tcw = inv(Twc)
    Tcw_matc = Mat4C(Tcw)
    # clear deepindexmap
    clear!(deep_index_map)
    # get index map
    for surfel_data in surfel_data_collections
        ccall(
            (:predict_index_map_wo_clear, :libsurfels),
            Cvoid,
            (
                SurfelsDataC,
                Clonglong,
                DeepIndexMapC,
                CUDA.CuPtr{Cint},
                Mat4C,
                CamParamsC,
                Cfloat,
                Cfloat,
                Cint,
                Cint,
            ),
            surfel_data.cstruct,
            surfel_data.count,
            deep_index_map.cstruct,
            d_label_predict_ptr,
            Tcw_matc,
            cam,
            Float32(max_depth),
            Float32(confidence_thres),
            Int32(timestamp),
            Int32(timedelta),
        )
    end
    # if !inplace
    # copy all buffers for not inplace return
    depth_map = deep_index_map.depth_map
    color_map = deep_index_map.color_map

    a_depth = collect(depth_map[1, :, :])
    invalid_mask = a_depth .> max_depth

    a_depth[invalid_mask] .= 0.0
    # depth_view = Gray.(a_depth)
    a_color = collect(color_map[1, :, :])

    a_img = reinterpret(Images.RGB4{Images.N0f8}, a_color)
    a_img[invalid_mask] .= Images.RGB4(0, 0, 0)

    return a_depth, a_img
    # else
    #       depth_map = deep_index_map.depth_map
    #       color_map = deep_index_map.color_map
    #       return depth_map, color_map
    # end

end


function render(
    surfel_data_collections::Union{
        Base.ValueIterator{Dict{Int64,GlSurfelsData}},
        Vector{GlSurfelsData},
    },
    Twc::Matrix{Float64},
    confidence_thres::Float64,
    timestamp::Int64,
    timedelta::Int64,
    program,
    frame_buffer::GLFrameBuffer;
    test_in_view::Bool=false
)
    in_view_count = 0
    # bind(frame_buffer)
    glCheckError()
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer.fbo)
    glCheckError()

    Tcw = inv(Twc)
    Tcw_matc = Mat4C(Tcw)

    # TODO this first_frag can be replaced with L617-L619
    first_frag = true

    # clear to avoid left over information used
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)#GL_DEPTH_BUFFER_BIT

    # get index map
    for gl_surfels_data in surfel_data_collections
        if test_in_view && (!isempty(gl_surfels_data.bbox_corners)) && (!bbox_in_view(gl_surfels_data, Tcw, frame_buffer.cam))
            # @show "not in view"
            continue
            # else
            # @show "in view"
        end
        in_view_count += 1

        release_cuda_ptr!(gl_surfels_data)
        if first_frag
            first_frag = false
            gl_render_surfels(
                program,
                gl_surfels_data.gl_vbo,
                gl_surfels_data.surfels_data.count,
                Tcw,
                frame_buffer.proj_mat,
                confidence_thres, #confidence_thres::Float64,
                timestamp, #timestamp::Int64,
                timedelta,
                clear = true,
            )#, timedelta::Int64;
        else
            gl_render_surfels(
                program,
                gl_surfels_data.gl_vbo,
                gl_surfels_data.surfels_data.count,
                Tcw,
                frame_buffer.proj_mat,
                confidence_thres, #confidence_thres::Float64,
                timestamp, #timestamp::Int64,
                timedelta,
                clear = false,
            )#, timedelta::Int64;
        end
    end
    glCheckError()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glCheckError()


    if (in_view_count == 0) || (first_frag == true)
        @warn "there is no surfels rendered! in: $(@__FILE__)"
    end

    return in_view_count
end


function observe(
    surfel_data_collections::Union{
        Base.ValueIterator{Dict{Int64,GlSurfelsData}},
        Vector{GlSurfelsData},
    },
    Twc::Matrix{Float64},
    # cam::CamParamsC,
    # max_depth::Float64,
    confidence_thres::Float64,
    timestamp::Int64,
    timedelta::Int64,
    program,
    frame_buffer::Surfels.GLFrameBuffer,
)

    render(
        surfel_data_collections,
        Twc,
        confidence_thres,
        timestamp,
        timedelta,
        program,
        frame_buffer,
    )

    a_img, a_depth = Surfels.read_gl_color_depth(frame_buffer)
    #(proj_mat, Int64(cam.width), Int64(cam.height))

    # a_depth = collect(depth_map[1, :, :])
    invalid_mask = a_depth .== 0.0

    # a_depth[invalid_mask] .= 0.0
    # depth_view = Gray.(a_depth)
    # a_color = collect(color_map[1, :, :])

    # a_img = reinterpret(Images.RGB4{Images.N0f8}, a_color)
    a_img[invalid_mask] .= Images.RGB4(0, 0, 0)

    return a_depth, a_img
end
