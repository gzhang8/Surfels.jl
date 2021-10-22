import Pangolin
using ModernGL
using GlHelper

mutable struct SurfelsViewer
    name::String
    s_cam::Pangolin.OpenGlRenderState
    handler::Pangolin.Handler3D
    d_cam::Pangolin.View
    render_gl_program::GLint
end

function SurfelsViewer(name::String = "Surfels Viewer")

    w = 640
    h = 480

    Pangolin.create_window(name, w, h)

    s_cam = Pangolin.OpenGlRenderState()
    handler = Pangolin.Handler3D(s_cam)
    d_cam = Pangolin.View(handler)


    # Create and initialize shaders
    # TODO: change how color is made

    root_dir = dirname(@__DIR__)

    vsh = Base.read(joinpath(root_dir, "gl/draw_global_surface.vert"), String)
    fsh = Base.read(joinpath(root_dir, "gl/draw_global_surface.frag"), String)
    geom_sh = Base.read(joinpath(root_dir, "gl/draw_global_surface.geom"), String)

    vertexShader = createShader(vsh, GL_VERTEX_SHADER)
    fragmentShader = createShader(fsh, GL_FRAGMENT_SHADER)
    geom_shader = createShader(geom_sh, GL_GEOMETRY_SHADER)

    program = createShaderProgram3(vertexShader, fragmentShader, geom_shader)

    res = SurfelsViewer(name, s_cam, handler, d_cam, program)
    return res
end


function make_current(sv::SurfelsViewer)
    res = Pangolin.make_window_context_current(sv.name)
    @assert res == true "make window current failed"
end

function show_surfels(
    sv::SurfelsViewer,
    surfels_vbo::GLuint,
    surfels_count::Union{Int64,UInt64};
    threshold::Float64 = 0.1,
    show_unstable::Bool = false,
    color_type::Int64 = 2,
    current_time::Int64 = 2,
    time_delta::Int64 = 100000,
    vertex_size::Int64 = 12,
    Twc::Matrix{Float64} = Matrix{Float64}(LinearAlgebra.I, 4, 4),
    make_viewer_current::Bool=true
)
    if (make_viewer_current)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glCheckError()
        make_current(sv)
        glCheckError()
    end

    # Generate a vertex array and array buffer for our data
    vao = glGenVertexArray()
    glBindVertexArray(vao)
    # vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, surfels_vbo)


    program = sv.render_gl_program
    glUseProgram(sv.render_gl_program)

    # proj_mat = ProjectionMatrixRDF_TopLeft(640, 480, 525.0, 525.0, 320.0, 240.0, 0.1, 1000.0)
    # # MVP is a float32 matrix
    # mvp64 = proj_mat * T_base_inv

    Pangolin.set_model_view_mat!(sv.s_cam, Twc)

    # mvp_init = convert(Matrix{Float32}, mvp64)
    # mvp_loc = glGetUniformLocation(program, "MVP")
    # # void glUniformMatrix4fv(	GLint location,
    # #  	GLsizei count,
    # #  	GLboolean transpose,
    # #  	const GLfloat *value);
    # glUniformMatrix4fv(mvp_loc, 1, false,  Base.unsafe_convert(Ptr{GLfloat}, mvp));

    threshold_loc = glGetUniformLocation(program, "threshold")
    glUniform1f(threshold_loc, threshold)

    unstable_loc = glGetUniformLocation(program, "unstable")
    glUniform1i(unstable_loc, Int(show_unstable))

    drawWindow_loc = glGetUniformLocation(program, "drawWindow")
    glUniform1i(drawWindow_loc, 0)

    color_type_loc = glGetUniformLocation(program, "colorType")
    glUniform1i(color_type_loc, color_type)

    time_loc = glGetUniformLocation(program, "time")
    glUniform1i(time_loc, current_time)

    time_delta_loc = glGetUniformLocation(program, "timeDelta")
    glUniform1i(time_delta_loc, time_delta)

    # glEnableVertexAttribArray(positionAttribute)
    # glVertexAttribPointer(positionAttribute, 2, GL_FLOAT, false, 0, C_NULL)
    positionAttribute = 0
    vertex_byte_size::Int64 = vertex_size * sizeof(GLfloat)
    glEnableVertexAttribArray(positionAttribute)
    glVertexAttribPointer(
        positionAttribute,
        4,
        GL_FLOAT,
        GL_FALSE,
        vertex_byte_size,
        C_NULL,
    )

    glEnableVertexAttribArray(positionAttribute + 1)
    glVertexAttribPointer(
        1,
        4,
        GL_FLOAT,
        GL_FALSE,
        vertex_byte_size,
        reinterpret(Ptr{GLCvoid}, Int64(sizeof(GLfloat) * 4)),
    )

    glEnableVertexAttribArray(positionAttribute + 2)
    glVertexAttribPointer(
        2,
        4,
        GL_FLOAT,
        GL_FALSE,
        vertex_byte_size,
        reinterpret(Ptr{GLCvoid}, Int64(sizeof(GLfloat) * 8)),
    )

    # TODO: add depth test for depth buffer and color when there is overlapping
    glEnable(GL_DEPTH_TEST)

    # fixed_view = false

    glCheckError()
    # Loop until the user closes the window
    while !Pangolin.should_quit()
        # Pulse the background blue
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)#GL_DEPTH_BUFFER_BIT
        # Draw our triangle

        # mvp = Pangolin.get_projection_view_mat(s_cam)
        # if fixed_view
        # 	# global first_time = false
        # 	Pangolin.set_model_view_mat!(s_cam, T_base)
        # else
        Pangolin.activate(sv.d_cam, sv.s_cam)
        # end
        mvp = Pangolin.get_projection_view_mat(sv.s_cam)
        mvp32 = convert(Matrix{Float32}, mvp)
        # mvp32 = convert(Matrix{Float32}, mvp64)
        mvp_loc = glGetUniformLocation(program, "MVP")
        # void glUniformMatrix4fv(	GLint location,
        #  	GLsizei count,
        #  	GLboolean transpose,
        #  	const GLfloat *value);
        glUniformMatrix4fv(mvp_loc, 1, false, Base.unsafe_convert(Ptr{GLfloat}, mvp32))

        glCheckError()
        glDrawArrays(GL_POINTS, 0, surfels_count)

        glCheckError()
        # Swap front and back buffers
        # GLFW.SwapBuffers(window)
        # # Poll for and process events
        # GLFW.PollEvents()
        Pangolin.finish_frame()
        glCheckError()
    end
    # GLFW.Terminate()

    # Pangolin.destroy_window("viewer")
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glDeleteVertexArrays(1, [vao])

    glCheckError()


end


function render(
    sv::SurfelsViewer,
    surfels_vbo::GLuint,
    surfels_count::Union{Int64,UInt64};
    threshold::Float64 = 0.1,
    show_unstable::Bool = false,
    color_type::Int64 = 2,
    current_time::Int64 = 2,
    time_delta::Int64 = 100000,
    vertex_size::Int64 = 12,
    view_from_Twc::Bool = false,
    Twc::Matrix{Float64} = Matrix{Float64}(LinearAlgebra.I, 4, 4),
    make_viewer_current::Bool=true
)
    if (make_viewer_current)
        # disable any app frame buffers
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glCheckError()
        make_current(cv)
        glCheckError()
    end

    # Generate a vertex array and array buffer for our data
    glCheckError("Debug check1")
    vao = glGenVertexArray()
    glBindVertexArray(vao)
    # vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, surfels_vbo)


    program = sv.render_gl_program
    glUseProgram(sv.render_gl_program)

    # proj_mat = ProjectionMatrixRDF_TopLeft(640, 480, 525.0, 525.0, 320.0, 240.0, 0.1, 1000.0)
    # # MVP is a float32 matrix
    # mvp64 = proj_mat * T_base_inv
    if view_from_Twc
        Pangolin.set_model_view_mat!(sv.s_cam, Twc)
    end

    # mvp_init = convert(Matrix{Float32}, mvp64)
    # mvp_loc = glGetUniformLocation(program, "MVP")
    # # void glUniformMatrix4fv(	GLint location,
    # #  	GLsizei count,
    # #  	GLboolean transpose,
    # #  	const GLfloat *value);
    # glUniformMatrix4fv(mvp_loc, 1, false,  Base.unsafe_convert(Ptr{GLfloat}, mvp));

    threshold_loc = glGetUniformLocation(program, "threshold")
    glUniform1f(threshold_loc, threshold)

    unstable_loc = glGetUniformLocation(program, "unstable")
    glUniform1i(unstable_loc, Int(show_unstable))

    drawWindow_loc = glGetUniformLocation(program, "drawWindow")
    glUniform1i(drawWindow_loc, 0)

    color_type_loc = glGetUniformLocation(program, "colorType")
    glUniform1i(color_type_loc, color_type)

    time_loc = glGetUniformLocation(program, "time")
    glUniform1i(time_loc, current_time)

    time_delta_loc = glGetUniformLocation(program, "timeDelta")
    glUniform1i(time_delta_loc, time_delta)

    # glEnableVertexAttribArray(positionAttribute)
    # glVertexAttribPointer(positionAttribute, 2, GL_FLOAT, false, 0, C_NULL)
    positionAttribute = 0
    vertex_byte_size::Int64 = vertex_size * sizeof(GLfloat)
    glEnableVertexAttribArray(positionAttribute)
    glVertexAttribPointer(
        positionAttribute,
        4,
        GL_FLOAT,
        GL_FALSE,
        vertex_byte_size,
        C_NULL,
    )

    glEnableVertexAttribArray(positionAttribute + 1)
    glVertexAttribPointer(
        1,
        4,
        GL_FLOAT,
        GL_FALSE,
        vertex_byte_size,
        reinterpret(Ptr{GLCvoid}, Int64(sizeof(GLfloat) * 4)),
    )

    glEnableVertexAttribArray(positionAttribute + 2)
    glVertexAttribPointer(
        2,
        4,
        GL_FLOAT,
        GL_FALSE,
        vertex_byte_size,
        reinterpret(Ptr{GLCvoid}, Int64(sizeof(GLfloat) * 8)),
    )

    # TODO: add depth test for depth buffer and color when there is overlapping
    glEnable(GL_DEPTH_TEST)

    # fixed_view = false

    # Loop until the user closes the window
    # while !Pangolin.should_quit()
    # Pulse the background blue

    # Draw our triangle

    # mvp = Pangolin.get_projection_view_mat(s_cam)
    # if fixed_view
    # 	# global first_time = false
    # 	Pangolin.set_model_view_mat!(s_cam, T_base)
    # else
    Pangolin.activate(sv.d_cam, sv.s_cam)
    # end
    mvp = Pangolin.get_projection_view_mat(sv.s_cam)
    mvp32 = convert(Matrix{Float32}, mvp)
    # mvp32 = convert(Matrix{Float32}, mvp64)
    mvp_loc = glGetUniformLocation(program, "MVP")
    # void glUniformMatrix4fv(	GLint location,
    #  	GLsizei count,
    #  	GLboolean transpose,
    #  	const GLfloat *value);
    glUniformMatrix4fv(mvp_loc, 1, false, Base.unsafe_convert(Ptr{GLfloat}, mvp32))


    glDrawArrays(GL_POINTS, 0, surfels_count)
    # Swap front and back buffers
    # GLFW.SwapBuffers(window)
    # # Poll for and process events
    # GLFW.PollEvents()

    # end
    # GLFW.Terminate()

    # Pangolin.destroy_window("viewer")
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glDeleteVertexArrays(1, [vao])

    glCheckError()


end

function render(
    sv::SurfelsViewer,
    o::GlSurfelsData;
    threshold::Float64 = 0.1,
    show_unstable::Bool = false,
    color_type::Int64 = 2,
    current_time::Int64 = 2,
    time_delta::Int64 = 100000,
    vertex_size::Int64 = 12,
    view_from_Twc::Bool = false,
    Twc::Matrix{Float64} = Matrix{Float64}(LinearAlgebra.I, 4, 4),
    make_viewer_current::Bool=true
)
    if (make_viewer_current)
        # disable any app frame buffers
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glCheckError()
        make_current(sv)
        glCheckError()
    end

    old_mapped2cuda = o.mapped2cuda
    if o.mapped2cuda
        release_cuda_ptr!(o)
    end

    glCheckError("Debug check2")

    surfels_vbo::GLuint = o.gl_vbo
    surfels_count = o.surfels_data.count
    render(
        sv,
        surfels_vbo,
        surfels_count,
        threshold = threshold,
        show_unstable = show_unstable,
        color_type = color_type,
        current_time = current_time,
        time_delta = time_delta,
        vertex_size = vertex_size,
        view_from_Twc = view_from_Twc,
        Twc = Twc,
        make_viewer_current=false # because it is current already
    )
    if old_mapped2cuda
        get_cuda_ptr!(o)
    end
end

function finish_frame(sv::SurfelsViewer)
    Pangolin.finish_frame()
end

function clear(sv::SurfelsViewer)
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)#GL_DEPTH_BUFFER_BIT
end

"""
Visualize GlSurfelsData while keep mapped to cuda unchanged
"""
function show_surfels(
    sv::SurfelsViewer,
    o::GlSurfelsData;
    threshold::Float64 = 0.1,
    show_unstable::Bool = false,
    color_type::Int64 = 2,
    current_time::Int64 = 2,
    time_delta::Int64 = 100000,
    vertex_size::Int64 = 12,
    Twc::Matrix{Float64} = Matrix{Float64}(LinearAlgebra.I, 4, 4),
    make_viewer_current::Bool=true
)
    if (make_viewer_current)
        # disable any app frame buffers
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glCheckError()
        make_current(sv)
        glCheckError()
    end

    old_mapped2cuda = o.mapped2cuda
    if o.mapped2cuda
        release_cuda_ptr!(o)
    end

    surfels_vbo::GLuint = o.gl_vbo
    surfels_count = o.surfels_data.count
    show_surfels(
        sv,
        surfels_vbo,
        surfels_count,
        threshold = threshold,
        show_unstable = show_unstable,
        color_type = color_type,
        current_time = current_time,
        time_delta = time_delta,
        vertex_size = vertex_size,
        Twc = Twc,
        make_viewer_current=false # because it is current already
    )
    if old_mapped2cuda
        get_cuda_ptr!(o)
    end
end

"""
Show a set of fragments
"""
function render(
    sv::SurfelsViewer,
    frags::Vector{GlSurfelsData};
    threshold::Float64 = 0.1,
    show_unstable::Bool = false,
    color_type::Int64 = 2,
    current_time::Int64 = 2,
    time_delta::Int64 = 100000,
    vertex_size::Int64 = 12,
    view_from_Twc::Bool = false,
    Twc::Matrix{Float64} = Matrix{Float64}(LinearAlgebra.I, 4, 4),
    make_viewer_current::Bool=true
)
    if (make_viewer_current)
        # disable any app frame buffers
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glCheckError()
        make_current(sv)
        glCheckError()
    end

    glCheckError("Debug check 4")

    # dd
    #
    for frag in frags
        glCheckError("Debug check 5")

        render(
            sv,
            frag,
            threshold = threshold,
            show_unstable = show_unstable,
            color_type = color_type,
            current_time = current_time,
            time_delta = time_delta,
            vertex_size = vertex_size,
            view_from_Twc = view_from_Twc,
            Twc = Twc,
            make_viewer_current=false # because it is current already
        )
    end
    # finish_frame(viewer)
end


function show_surfels(
    sv::SurfelsViewer,
    frags::Vector{GlSurfelsData};
    threshold::Float64 = 0.1,
    show_unstable::Bool = false,
    color_type::Int64 = 2,
    current_time::Int64 = 2,
    time_delta::Int64 = 100000,
    vertex_size::Int64 = 12,
    Twc::Matrix{Float64} = Matrix{Float64}(LinearAlgebra.I, 4, 4),
    make_viewer_current::Bool=true
)
    if (make_viewer_current)
        # disable any app frame buffers
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glCheckError()
        make_current(sv)
        glCheckError()
    end

    glCheckError("Debug check 7")

    while !Pangolin.should_quit()
        glCheckError("Debug check6")
        clear(sv)
        glCheckError("Debug check7")
        render(
            sv,
            frags,
            threshold = threshold,
            show_unstable = show_unstable,
            color_type = color_type,
            current_time = current_time,
            time_delta = time_delta,
            vertex_size = vertex_size,
            view_from_Twc = false, # this must be false, we set Twc to s_cam
            Twc = Twc,
            make_viewer_current=false # because it is current already
        )
        finish_frame(sv)
    end
end
