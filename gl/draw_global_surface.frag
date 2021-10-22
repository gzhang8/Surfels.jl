#version 330 core

in vec3 vColor0;
in vec2 texcoord;
in float radius;
flat in int unstablePoint;

out vec4 FragColor;

void main()
{
    if(dot(texcoord, texcoord) > 1.0)
        discard;

    FragColor = vec4(vColor0, 1.0f);

    if(unstablePoint == 1)
	{
		gl_FragDepth = gl_FragCoord.z + radius;
	}
    else
   	{
   		gl_FragDepth = gl_FragCoord.z;
   	}
}
