import GLFW
using ModernGL
# using Surfels
using Images
using LinearAlgebra

using GlHelper


mutable struct GLFrameBuffer
	fbo::GLuint
	color_render_buf::GLuint
	depth_render_buf::GLuint
	depth_2nd_tex::GLuint
	color_cuda_res::Vector{Ptr{Cvoid}}
	depth_cuda_res::Vector{Ptr{Cvoid}}
	a::Float64 # abc are for recovering real depth form z buffer
	b::Float64
	c::Float64
	min_depth::Float64
	max_depth::Float64
	proj_mat::Matrix{Float64}
	cam::CameraModel.RgbdCamParams
end

function GLFrameBuffer(w::Int64, h::Int64, cam::CameraModel.RgbdCamParams,
	                   min_depth::Float64, max_depth::Float64;
					   use_texture::Bool=false)
    # proj_mat = ProjectionMatrixRDF_TopLeft(Int64(cam.width), Int64(cam.height),
	#                                        Float64(cam.fx), Float64(cam.fy),
	# 									   Float64(cam.cx), Float64(cam.cy),
	# 									   min_depth, max_depth*1.01)
	proj_mat = gl_projection_matrix(cam, min_depth, max_depth*1.01)
	a = proj_mat[3, 3]
   	b = proj_mat[3, 4]
   	c = proj_mat[4, 3]

	# GLuint fbo, render_buf;
	fbo = GLuint[0]
	render_buf = GLuint[0, 0] # rand value
	glGenFramebuffers(1, fbo);
	if use_texture
		glGenRenderbuffers(1, render_buf);
	else
		glGenRenderbuffers(2, render_buf);
	end
	# allocate memory for color
	glBindRenderbuffer(GL_RENDERBUFFER, render_buf[1]);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1])
	# attach the color renderbuffer object
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, render_buf[1]);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

    depthTextureID = GLuint[0, 0]
	color_cuda_res = Ptr{Cvoid}[C_NULL]
	depth_cuda_res = Ptr{Cvoid}[C_NULL]
    if use_texture
		@show "use texture as depth render buffer"
        glCheckError()
		glGenTextures(2, depthTextureID);
		glBindTexture(GL_TEXTURE_2D, depthTextureID[1]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glCheckError()

		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, C_NULL);
		glCheckError()
		# attach tex to the framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1])
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTextureID[1], 0);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
		glCheckError()


		# void register_gl_image(GLuint gl_image, void**cuda_res_ptr);
		# TODO: cuda doesn't support depth buffer from gl
		# https://stackoverflow.com/questions/5904998/support-of-gl-depth-component-for-cudagraphicsglregisterimage?answertab=active#tab-top
		# see https://community.khronos.org/t/cuda-interop-for-depth-component/72860/2

		# fastest with gl > 4.3 https://gamedev.stackexchange.com/questions/108194/whats-the-fastest-way-to-copy-a-texture-to-another-texture-in-opengl
		glBindTexture(GL_TEXTURE_2D, depthTextureID[2]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glCheckError()
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, C_NULL);
		glCheckError()

		ccall((:register_gl_image, :libsurfels),
		      Cvoid,
			  (UInt32, Ptr{Ptr{Cvoid}}),
			  depthTextureID[2], Base.unsafe_convert(Ptr{Ptr{Cvoid}}, depth_cuda_res))
	    @show depth_cuda_res
	else
		# allocate memory for depth
		glBindRenderbuffer(GL_RENDERBUFFER, render_buf[2]);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		# attach to framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1])
		glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_buf[2]);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
	end


	if use_texture
		f_bufs = GLFrameBuffer(fbo[1], render_buf[1], depthTextureID[1],
							   depthTextureID[2], color_cuda_res, depth_cuda_res,
							   a, b, c, min_depth, max_depth, proj_mat, cam)
		finalizer(free_GLFrameBuffer_tex_buffer, f_bufs)
		return f_bufs
	else
		f_bufs = GLFrameBuffer(fbo[1], render_buf[1], render_buf[2],
							   depthTextureID[2], color_cuda_res, depth_cuda_res,
							   a, b, c, min_depth, max_depth, proj_mat, cam)
		finalizer(free_GLFrameBuffer_render_buffer, f_bufs)
		return f_bufs
	end

end

function free_GLFrameBuffer_render_buffer(f_bufs::GLFrameBuffer)
	glDeleteRenderbuffers(2, [f_bufs.color_render_buf, f_bufs.depth_render_buf])
	glDeleteFramebuffers(1, [f_bufs.fbo])
end

function free_GLFrameBuffer_tex_buffer(f_bufs::GLFrameBuffer)
	glDeleteRenderbuffers(1, [f_bufs.color_render_buf])
	glDeleteTextures(1, [f_bufs.depth_render_buf])
	#  unmap and delete cuda_res
	if f_bufs.depth_cuda_res[1] != C_NULL
		ccall((:unregister_gl_cuda, :libsurfels),
	          Cvoid, (Ptr{Cvoid}, ),
	          f_bufs.depth_cuda_res[1])
	end
	glDeleteTextures(1, [f_bufs.depth_2nd_tex])
	glDeleteFramebuffers(1, [f_bufs.fbo])
end

function bind(o::GLFrameBuffer)
	glCheckError()
	glBindFramebuffer(GL_FRAMEBUFFER, o.fbo)
	glCheckError()
end

function gl_render_surfels(program,
	                       surfel_data_vbo::GLuint,
	                       count::Int64,
	                       Tcw::Matrix{Float64},
						   proj_mat::Matrix{Float64},
						   # max_depth::Float64,
	                       confidence_thres::Float64,
	                       timestamp::Int64,
	                       timedelta::Int64;
						   draw_unstable::Bool=false,
						   clear::Bool=true)

	glCheckError()
	glBindBuffer(GL_ARRAY_BUFFER, surfel_data_vbo);
	glCheckError()
	glUseProgram(program)
	glCheckError()
	# MVP is a float32 matrix
	mvp64 = proj_mat * Tcw
	mvp = convert(Matrix{Float32}, mvp64)
	mvp_loc = glGetUniformLocation(program, "MVP")
	# void glUniformMatrix4fv(	GLint location,
	#  	GLsizei count,
	#  	GLboolean transpose,
	#  	const GLfloat *value);
	glUniformMatrix4fv(mvp_loc, 1, false,  Base.unsafe_convert(Ptr{GLfloat}, mvp));
    glCheckError()
	threshold_loc = glGetUniformLocation(program, "threshold")
	glUniform1f(threshold_loc, confidence_thres);

	unstable_loc = glGetUniformLocation(program, "unstable")
	if draw_unstable
	    glUniform1i(unstable_loc, 1);
	else
		glUniform1i(unstable_loc, 0);
	end
	glCheckError()
	# TODO: what is this?
	drawWindow_loc = glGetUniformLocation(program, "drawWindow")
	glUniform1i(drawWindow_loc, 0);

	color_type_loc = glGetUniformLocation(program, "colorType")
	glUniform1i(color_type_loc, 2)

	time_loc = glGetUniformLocation(program, "time")
	glUniform1i(time_loc, timestamp)

	time_delta_loc = glGetUniformLocation(program, "timeDelta")
	glUniform1i(time_delta_loc, timedelta)

    glCheckError()
	positionAttribute = 0
	Vertex_SIZE = 12 * sizeof(GLfloat)
	glEnableVertexAttribArray(positionAttribute);
	glCheckError()
	glVertexAttribPointer(positionAttribute, 4, GL_FLOAT, GL_FALSE, Vertex_SIZE, C_NULL);
    glCheckError()

	glEnableVertexAttribArray(positionAttribute+1);
	glVertexAttribPointer(positionAttribute+1, 4, GL_FLOAT, GL_FALSE,
	                    Vertex_SIZE, reinterpret(Ptr{GLCvoid}, Int64(sizeof(GLfloat) * 4)));
    glCheckError()

	glEnableVertexAttribArray(positionAttribute+2);
	glVertexAttribPointer(positionAttribute+2, 4, GL_FLOAT, GL_FALSE,
	                    Vertex_SIZE, reinterpret(Ptr{GLCvoid}, Int64(sizeof(GLfloat) * 8)));

	glEnable(GL_DEPTH_TEST);
	if clear
		glClearColor(0.2, 0.3, 0.3, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)#GL_DEPTH_BUFFER_BIT
    end

	glCheckError()

	# Draw our points
	glDrawArrays(GL_POINTS, 0, count);

    glCheckError()
	glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
	glCheckError()

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    ModernGL.glFinish()
end


function read_gl_color_depth(frame_buffer::GLFrameBuffer)
	proj_mat = frame_buffer.proj_mat
	w = frame_buffer.cam.width
	h = frame_buffer.cam.height
	max_depth = frame_buffer.max_depth
	# proj_mat::Matrix{Float64}, w::Int64, h::Int64;
	#                          max_depth::Float64=10.0)
	# read data
	glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer.fbo)

	glReadBuffer(GL_COLOR_ATTACHMENT0);
	rgb_data = Vector{UInt8}(undef, 4 * h * w)
	# data is in row major order

	glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgb_data);
	# // Return to onscreen rendering:
	# glBindFramebuffer(GL_DRAW_FRAMEBUFFER​,0);

	# read depth
	# glReadBuffer(GL_DEPTH_ATTACHMENT)
	depth32 = Vector{Float32}(undef, h * w)
	glReadPixels(0, 0 , w, h, GL_DEPTH_COMPONENT, GL_FLOAT, depth32);

	a = proj_mat[3, 3]
	b = proj_mat[3, 4]
	c = proj_mat[4, 3]
	real_depth = b ./(c .* (2 .* depth32 .- 1.0) .- a)

	img = Matrix{RGB}(undef, h, w)
	depth = zeros(Float64, h, w)
	# glReadPixels and glReadnPixels return values from each pixel with lower left
	# corner at (x+i,y+j) for 0<=i<width and 0<=j<height. This pixel is said to be
	# the ith pixel in the jth row. Pixels are returned in row order from the lowest
	# to the highest row, left to right in each row.
	for i = 1:h
		for j = 1:w
			r = rgb_data[1 +  ((h - i) * w + j - 1) * 4] /255.0
			g = rgb_data[2 +  ((h - i) * w + j - 1) * 4] /255.0
			b = rgb_data[3 +  ((h - i) * w + j - 1) * 4] /255.0
			img[i, j] = RGB(r, g, b)
			d_value = real_depth[(h - i) * w + j]
			if d_value <= max_depth
				depth[i, j] = d_value
			end
		end
	end
	return img, depth
end


function read_gl_color(frame_buffer::GLFrameBuffer)
	# proj_mat = frame_buffer.proj_mat
	w = frame_buffer.cam.width
	h = frame_buffer.cam.height
	# max_depth = frame_buffer.max_depth
	# proj_mat::Matrix{Float64}, w::Int64, h::Int64;
	#                          max_depth::Float64=10.0)
	# read data
	glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer.fbo)
	# read data
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	rgb_data = Vector{UInt8}(undef, 4 * h * w)
	# data is in row major order

	glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgb_data);
	# // Return to onscreen rendering:
	# glBindFramebuffer(GL_DRAW_FRAMEBUFFER​,0);

	# read depth
	# glReadBuffer(GL_DEPTH_ATTACHMENT)

	img = Matrix{RGB}(undef, h, w)

	# glReadPixels and glReadnPixels return values from each pixel with lower left
	# corner at (x+i,y+j) for 0<=i<width and 0<=j<height. This pixel is said to be
	# the ith pixel in the jth row. Pixels are returned in row order from the lowest
	# to the highest row, left to right in each row.
	for i = 1:h
		for j = 1:w
			r = rgb_data[1 +  ((h - i) * w + j - 1) * 4] /255.0
			g = rgb_data[2 +  ((h - i) * w + j - 1) * 4] /255.0
			b = rgb_data[3 +  ((h - i) * w + j - 1) * 4] /255.0
			img[i, j] = RGB(r, g, b)
		end
	end
	return img
end

function gl_check_in_view(frame_buffer::GLFrameBuffer;
	                      r_data::Vector{UInt8}=Vector{UInt8}(undef,
						      frame_buffer.cam.height * frame_buffer.cam.width))
	# proj_mat = frame_buffer.proj_mat
	w = frame_buffer.cam.width
	h = frame_buffer.cam.height
	# max_depth = frame_buffer.max_depth
	# proj_mat::Matrix{Float64}, w::Int64, h::Int64;
	#                          max_depth::Float64=10.0)
	# read data
	glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer.fbo)

	glReadBuffer(GL_COLOR_ATTACHMENT0);
	# rgb_data = Vector{UInt8}(undef, h * w)
	# data is in row major order

	glReadPixels(0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, r_data);
	# for v =
	return any(x->x>0, r_data)
end



function create_hidden_gl_window(w::Int64, h::Int64)
	GLFW.Init()
	# OS X-specific GLFW hints to initialize the correct version of OpenGL

	# Create a windowed mode window and its OpenGL context
	GLFW.WindowHint(GLFW.VISIBLE, false)
	# glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	window = GLFW.CreateWindow(w, h, "")#, C_NULL, C_NULL
	# Make the window's context current
	GLFW.MakeContextCurrent(window)
	# GLFW.ShowWindow(window)
	# GLFW.SetWindowSize(window, w, h) # Seems to be necessary to guarantee that window > 0
	glViewport(0, 0, w, h)

	finalizer(GLFW.DestroyWindow, window)
	return window
end


function create_gl_render_program()
	# Create and initialize shaders
	base_folder = dirname(dirname(@__FILE__))
	vsh = Base.read(joinpath(base_folder, "gl/draw_global_surface.vert"), String)
	fsh = Base.read(joinpath(base_folder, "gl/draw_global_surface.frag"), String)
	geom_sh = Base.read(joinpath(base_folder, "gl/draw_global_surface.geom"), String)

	vertexShader = createShader(vsh, GL_VERTEX_SHADER)
	fragmentShader = createShader(fsh, GL_FRAGMENT_SHADER)
	geom_shader = createShader(geom_sh, GL_GEOMETRY_SHADER)

	program = createShaderProgram3(vertexShader, fragmentShader, geom_shader)
	return program
end

function create_convex_hull_program()
	# create program
	base_folder = dirname(dirname(@__FILE__))
	vsh = Base.read(joinpath(base_folder, "gl/convex_hull.vert"), String)
	fsh = Base.read(joinpath(base_folder, "gl/convex_hull.frag"), String)
	# geom_sh = Base.read(joinpath(base_folder, "gl/draw_global_surface.geom"), String)

	vertexShader = createShader(vsh, GL_VERTEX_SHADER)
	fragmentShader = createShader(fsh, GL_FRAGMENT_SHADER)
	# geom_shader = createShader(geom_sh, GL_GEOMETRY_SHADER)

	program = createShaderProgram2(vertexShader, fragmentShader)
	return program
end


function gl_render_convex_hull(program,
	                           gl_surfels::GlSurfelsData,
		                       Tcw::Matrix{Float64},
							   proj_mat::Matrix{Float64},
							   clear::Bool=true)

	glCheckError()
	glBindBuffer(GL_ARRAY_BUFFER, gl_surfels.gl_vbo);
	glCheckError()
	glUseProgram(program)
	glCheckError()
	# MVP is a float32 matrix
	mvp64 = proj_mat * Tcw
	mvp = convert(Matrix{Float32}, mvp64)
	mvp_loc = glGetUniformLocation(program, "MVP")
	# void glUniformMatrix4fv(	GLint location,
	#  	GLsizei count,
	#  	GLboolean transpose,
	#  	const GLfloat *value);
	glUniformMatrix4fv(mvp_loc, 1, false,  Base.unsafe_convert(Ptr{GLfloat}, mvp));
    glCheckError()


    glCheckError()
	positionAttribute = 0
	Vertex_SIZE = 12 * sizeof(GLfloat)
	glEnableVertexAttribArray(positionAttribute);
	glVertexAttribPointer(positionAttribute, 4, GL_FLOAT, GL_FALSE, Vertex_SIZE, C_NULL);
    glCheckError()


	if clear
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT)#GL_DEPTH_BUFFER_BIT
    end

	glCheckError()

	# Draw our points
	# glDrawArrays(GL_POINTS, 0, count);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl_surfels.gl_veo);

	# // Draw the triangles !
	glDrawElements(
		GL_TRIANGLES,      #// mode
		gl_surfels.n_facets,    #// count
		GL_UNSIGNED_INT,   #// type
		C_NULL           #// element array buffer offset
	)

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glCheckError()
	glDisableVertexAttribArray(0);

	glCheckError()
    glBindBuffer(GL_ARRAY_BUFFER, 0);

	ModernGL.glFinish()

end
