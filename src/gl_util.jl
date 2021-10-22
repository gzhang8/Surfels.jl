
function gl_projection_matrix(cam::CameraModel.RgbdCamParams, min_depth::Float64, max_depth::Float64)
    proj_mat = ProjectionMatrixRDF_TopLeft(
        Int64(cam.width),
        Int64(cam.height),
        Float64(cam.fx),
        Float64(cam.fy),
        Float64(cam.cx),
        Float64(cam.cy),
        min_depth,
        max_depth * 1.001,
    )
end

# proj_mat = ProjectionMatrixRDF_TopLeft(640,480,525.0,525.0,320.0,240.0,0.1,100.0)
