module Surfels


using CUDA
import CameraModel


export FrameMerger, DeepIndexMap, RemoveIndex, SurfelsUtil, SurfelsData,
       # CameraParameters,# data types: aka data holders

       # functions
       merge_first_frame!, merge_addtional_frame!, merge_frame!,
       predict_label!,

       # transformation related functions
       applyT!, observe,

       #IO related functions
       save, read_surfels_file,

       # Memory related functions
       copy, deepcopy, copyto!,

       show_surfels, clear, render, finish_frame,

       GlSurfelsData, get_cuda_ptr!, release_cuda_ptr!,

       # viewer
       viewer,

       # util
       get_xyz,

       icp

using SurfelsLib_jll


include("types.jl")
include("gl_util.jl")

include("gl_render.jl")
include("surfels_ops.jl")

# util
# include("map_viewer.jl")
include("surfels_viewer.jl")

include("io.jl")

include("util.jl")
# include("icp.jl")

# global variable for viewer
gloabl_viewer = nothing
function viewer()
    if gloabl_viewer == nothing
        global gloabl_viewer = SurfelsViewer()
        return gloabl_viewer
    else
        return gloabl_viewer
    end
end

end # module
