# import Open3D
import Images

function save(surfels_data::SurfelsData, ply_fname::String; conf_low::Float64 = 0.1)
    surfels_data_cpu = collect(surfels_data.data[1:surfels_data.count*12])

    ccall(
        (:SavePly, :libsurfels),
        Cint,
        (Cstring, Ptr{Cfloat}, Clonglong, Cfloat),
        ply_fname,
        Base.unsafe_convert(Ptr{Cfloat}, surfels_data_cpu),
        surfels_data.count,
        Float32(conf_low),
    )

end

function save(surfels_data::GlSurfelsData, ply_fname::String; conf_low::Float64 = 0.1)
    get_cuda_ptr!(surfels_data)
    save(surfels_data.surfels_data, ply_fname, conf_low = conf_low)
end

# uint64_t ReadSurfelsFile(char* file_name_c, float** vertex_data_out, int b_has_time)
function read_surfels_file(ply_fname::String; has_time::Bool)
    data_ptr_array = Vector{Ptr{Float32}}(undef, 1)
    data_ptr_array[1] = convert(Ptr{Float32}, 0)
    data_ptr_ptr = Base.unsafe_convert(Ptr{Ptr{Cfloat}}, data_ptr_array)
    count = ccall(
        (:ReadSurfelsFile, :libsurfels),
        Culonglong,
        (Cstring, Ptr{Ptr{Cfloat}}, Cint),
        ply_fname,
        data_ptr_ptr,
        Cint(has_time)
    )
    # wrap data to be managed by julia
    surfel_data_cpu = unsafe_wrap(Array, data_ptr_array[1], count * 12, own = true)
    return surfel_data_cpu, count
end

function read_surfels_file2cuda(ply_fname::String; has_time::Bool=true)
    surfel_data_cpu, count = read_surfels_file(ply_fname, has_time=has_time)
    surfel_data = SurfelsData(surfel_data_cpu, Int64(count))
    return surfel_data
end

function read_surfels_file2gl(
    ply_fname::String;
    build_convex_hull::Bool = true,
    build_bbox::Bool = true,
    has_time::Bool=true)
    surfel_data_cpu, count = read_surfels_file(ply_fname, has_time=has_time)
    gl_surfel_data = GlSurfelsData(
        surfel_data_cpu,
        Int64(count),
        build_convex_hull=build_convex_hull,
        build_bbox=build_bbox)
    return gl_surfel_data
end

# function read_pcd2cpu(pcd_fname::String)

#     pcd_o3d = Open3D.read_point_cloud(pcd_fname)

#     xyz = Open3D.get_xyz_from_o3d(pcd_o3d)
#     normal = Open3D.get_normal_from_o3d(pcd_o3d)
#     rgb = Open3D.get_rgb_from_o3d(pcd_o3d) #.* 255

#     confidence = 10
#     init_time = 1.0

#     ## save data from arrays to surfels
#     # import Surfels
#     # import Images

#     # create cpu array for data mapping
#     surfel_count = size(xyz, 2)#Int64(1 * 1024^3 / 4)
#     surfels_data_cpu = Vector{Float32}(undef, surfel_count * 12)
#     # point_count = size(xyz, 2)

#     for idx = 1:surfel_count
#         #     // 0: x, 1 y, 2 z, 3 confidence, 4  color, 5 label , 6 init time, 7 update time,
#         #     // 8 nx 9 ny 10 nz, 11 radius

#         surfels_data_cpu[(idx-1)*12+1] = xyz[1, idx]
#         surfels_data_cpu[(idx-1)*12+2] = xyz[2, idx]
#         surfels_data_cpu[(idx-1)*12+3] = xyz[3, idx]
#         surfels_data_cpu[(idx-1)*12+4] = confidence
#         a_rgb = [Images.RGB4{Images.N0f8}(rgb[1, idx], rgb[2, idx], rgb[3, idx])]
#         surfels_data_cpu[(idx-1)*12+5] = reinterpret(Float32, a_rgb)[1]
#         surfels_data_cpu[(idx-1)*12+6] = 1.0
#         surfels_data_cpu[(idx-1)*12+7] = 1.0
#         surfels_data_cpu[(idx-1)*12+8] = 1.0
#         surfels_data_cpu[(idx-1)*12+9] = normal[1, idx]
#         surfels_data_cpu[(idx-1)*12+10] = normal[2, idx]
#         surfels_data_cpu[(idx-1)*12+11] = normal[3, idx]
#         surfels_data_cpu[(idx-1)*12+12] = 0.01 #11 radius
#     end

#     # get 2 GB for surfels

#     # surfel_data = Surfels.SurfelsData(surfel_count)

#     # upload to cuarray
#     # copyto!(surfel_data.data, surfels_data_cpu)
#     # surfel_data.count=surfel_count
#     return surfels_data_cpu, surfel_count
# end


# function read_pcd2cuda(pcd_fname::String)
#     surfel_data_cpu, count = read_pcd2cpu(pcd_fname)
#     surfel_data = SurfelsData(surfel_data_cpu, Int64(count))
#     return surfel_data
# end


# function read_pcd2gl(pcd_fname::String)
#     surfel_data_cpu, count = read_pcd2cpu(pcd_fname)
#     gl_surfel_data = GlSurfelsData(surfel_data_cpu, Int64(count))
#     return gl_surfel_data
# end
