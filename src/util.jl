function get_xyz(surfels::Surfels.SurfelsData)
    cpu_data = collect(surfels.data[1:surfels.count*12])
    xyz = Matrix{Float32}(undef, 3, surfels.count)
    for i = 1:surfels.count
        xyz[1, i] = cpu_data[(i-1)*12+1]
        xyz[2, i] = cpu_data[(i-1)*12+2]
        xyz[3, i] = cpu_data[(i-1)*12+3]
    end
    return xyz
end

function get_xyz(surfels::Surfels.GlSurfelsData)
    cuda = surfels.mapped2cuda
    if !cuda
        get_cuda_ptr!(surfels)
    end
    xyz = get_xyz(surfels.surfels_data)
    if !cuda
        release_cuda_ptr!(surfels)
    end
    return xyz
end
