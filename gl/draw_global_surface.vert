#version 330 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 normal;

uniform mat4 MVP;
uniform float threshold;
uniform int colorType;
uniform int unstable;
uniform int drawWindow;
uniform int time;
uniform int timeDelta;

out vec4 vColor;
out vec4 vPosition;
out vec4 vNormRad;
out mat4 vMVP;
out int vTime;
out int colorType0;
out int drawWindow0;
out int timeDelta0;

void main()
{
    if(position.w > threshold || unstable == 1)
    {
        colorType0 = colorType;
        drawWindow0 = drawWindow;
	    vColor = color;
	    vPosition = position;
	    vNormRad = normal;
	    vMVP = MVP;
	    vTime = time;
	    timeDelta0 = timeDelta;
	    gl_Position = MVP * vec4(position.xyz, 1.0);
    }
    else
    {
        colorType0 = -1;
    }
}
