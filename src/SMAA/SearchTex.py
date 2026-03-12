
from PIL import Image

def lerp(v0, v1, p):
    return v0 + (v1 - v0) * p

def bilinear(e):
    # e[0]       e[1]
    #
    #          x <-------- Sample position:    (-0.25,-0.125)
    # e[2]       e[3] <--- Current pixel [3]:  (  0.0, 0.0  )
    a = lerp(e[0], e[1], 1.0 - 0.25)
    b = lerp(e[2], e[3], 1.0 - 0.25)
    return lerp(a, b, 1.0 - 0.125)

edge = {
    bilinear([0, 0, 0, 0]): [0, 0, 0, 0],
    bilinear([0, 0, 0, 1]): [0, 0, 0, 1],
    bilinear([0, 0, 1, 0]): [0, 0, 1, 0],
    bilinear([0, 0, 1, 1]): [0, 0, 1, 1],

    bilinear([0, 1, 0, 0]): [0, 1, 0, 0],
    bilinear([0, 1, 0, 1]): [0, 1, 0, 1],
    bilinear([0, 1, 1, 0]): [0, 1, 1, 0],
    bilinear([0, 1, 1, 1]): [0, 1, 1, 1],

    bilinear([1, 0, 0, 0]): [1, 0, 0, 0],
    bilinear([1, 0, 0, 1]): [1, 0, 0, 1],
    bilinear([1, 0, 1, 0]): [1, 0, 1, 0],
    bilinear([1, 0, 1, 1]): [1, 0, 1, 1],

    bilinear([1, 1, 0, 0]): [1, 1, 0, 0],
    bilinear([1, 1, 0, 1]): [1, 1, 0, 1],
    bilinear([1, 1, 1, 0]): [1, 1, 1, 0],
    bilinear([1, 1, 1, 1]): [1, 1, 1, 1],
}

def deltaLeft(left, top):
    d = 0

    if top[3] == 1:
        d += 1

    if d == 1 and top[2] == 1 and left[1] != 1 and left[3] != 1:
        d += 1

    return d

def deltaRight(left, top):
    d = 0

    if top[3] == 1 and left[1] != 1 and left[3] != 1:
        d += 1

    if d == 1 and top[2] == 1 and left[0] != 1 and left[2] != 1:
        d += 1

    return d

def debug(dir, texcoord, val, left, top):
    print(dir, texcoord, val)
    print("|%s %s| |%s %s|" % (left[0], left[1], top[0], top[1]))
    print("|%s %s| |%s %s|" % (left[2], left[3], top[2], top[3]))
    print()

def hook_format(image, name):
    with open(f'{name}.hook', 'w') as f:
        f.write(f'//!TEXTURE {name}\n')
        f.write(f'//!SIZE {image.size[0]} {image.size[1]}\n')
        f.write('//!FORMAT r8\n')
        f.write('//!FILTER NEAREST\n')
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                val = image.getpixel((x, y))
                f.write(f'{val:02x}')
            f.write('\n')

image = Image.new("L", (66, 33))
for x in range(33):
    for y in range(33):
        texcoord = 0.03125 * x, 0.03125 * y
        if texcoord[0] in edge and texcoord[1] in edge:
            edges = edge[texcoord[0]], edge[texcoord[1]]
            val = 127 * deltaLeft(*edges) # Maximize dynamic range to help compression
            image.putpixel((x, y), val)
            #debug("left: ", texcoord, val, *edges)

for x in range(33):
    for y in range(33):
        texcoord = 0.03125 * x, 0.03125 * y
        if texcoord[0] in edge and texcoord[1] in edge:
            edges = edge[texcoord[0]], edge[texcoord[1]]
            val = 127 * deltaRight(*edges) # Maximize dynamic range to help compression
            image.putpixel((33 + x, y), val)
            #debug("right: ", texcoord, val, *edges)

image = image.crop([0, 17, 64, 33])
image = image.transpose(Image.FLIP_TOP_BOTTOM)

image.save("SearchTex.tga")
hook_format(image, "SEARCH_TEX")

