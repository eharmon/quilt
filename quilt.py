#!/usr/bin/python

"""

Quilt.

Image tiler for OpenLayers and Google Maps designed for large input sets

Copyright (C) 2012 Eric Harmon

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from PIL import Image
import sys
import os
import math
import argparse
import json

def tile(tilesize, imagelist, prefix, debug, crop):
    """Generate a list of tiles given a list of images and metadata"""

    print "Generating %d sized tiles to %s from %d images" % (tilesize, prefix, len(imagelist))

    print "Determining overall landscape size..."

    # Let's calculate the size of our full image scape
    im = Image.open(imagelist[0][1])
    if(crop):
        im = im.crop((crop[0], crop[1], crop[2], crop[3]))
    size = im.size
    
    # Free the image for GC
    del im

    full_x_size = size[0]
    full_y_size = size[1]

    # Find how many times we move across the scape and do some trickery to reduce that
    # Add ones for the first image
    y_moves = len(filter(lambda x: x[0] == 'd', imagelist))
    # A y move is also an x move, add both together (note that a single image tile should simply have a 'd' command
    x_moves = len(filter(lambda x: x[0] == 'r', imagelist)) + y_moves
    x_full_landscape = int(size[0] * (1.0 * x_moves / y_moves))
    y_full_landscape = size[1] * y_moves

    print "Total landscape size: %d x %d" % (x_full_landscape, y_full_landscape)

    square_full_landscape = x_full_landscape if x_full_landscape > y_full_landscape else y_full_landscape

    print "Square landscape size: %d x %d" % (square_full_landscape, square_full_landscape)

    print "Determining maximum zoom level..."

    # Essentially zoom levels are 2^(zoom level) tiles per side. This means we need to see what the closest power of 2 to the number of tiles we need is.
    estimated_tiles_per_side = square_full_landscape / tilesize
    # Reverse the power of 2 with a log
    max_zoom = int(math.ceil(math.log(estimated_tiles_per_side, 2)))

    print "Maximum zoom level is %d" % max_zoom

    print "Pregenerating source image tilemap..."

    print "Determining start offsets..."

    xoffset = (1.0 * square_full_landscape - x_full_landscape) / 2
    yoffset = (1.0 * square_full_landscape - y_full_landscape) / 2

    print "Offsets %d x %d" % (xoffset, yoffset)

    # Subtacting 1 from max_zoom and computing a scale factor to offset since we're disallowing maximum zoom currently
    # TODO: Need to figure out how to more properly feed this data to openLayers
    tiles_per_side_for_zoom = math.pow(2, max_zoom)
    tiles_for_zoom = tiles_per_side_for_zoom * tiles_per_side_for_zoom
    square_for_zoom = tiles_per_side_for_zoom * tilesize
    scale_factor = (square_for_zoom / square_full_landscape)

    write_settings(max_zoom - 1, square_full_landscape * scale_factor, prefix)

    # This should be max_zoom + 1, disabled temporarily since the images are too huge
    for zoom in range(0, max_zoom):
#    for zoom in range(0, 2):

        print "\n\nGenerating zoom level %d" % zoom

        tiles_per_side_for_zoom = math.pow(2, zoom)
        tiles_for_zoom = tiles_per_side_for_zoom * tiles_per_side_for_zoom
        square_for_zoom = tiles_per_side_for_zoom * tilesize

        print "Will generate %d tiles at landscape size %d x %d" % (tiles_for_zoom, square_for_zoom, square_for_zoom)

        print "Determining zoomed landscape size..."

        scale_factor = (square_for_zoom / square_full_landscape)

        zoom_landscape_size_x = scale_factor * x_full_landscape
        zoom_landscape_size_y = scale_factor * y_full_landscape

        print "Zoomed landscape size: %d x %d" % (zoom_landscape_size_x, zoom_landscape_size_y)

        zoom_xoffset = scale_factor * xoffset
        zoom_yoffset = scale_factor * yoffset

        print "Zoomed offsets %d x %d" % (zoom_xoffset, zoom_yoffset)

        zoom_x_size = scale_factor * full_x_size
        zoom_y_size = scale_factor * full_y_size

        print "Generating input file tilemap"

        input_tilemap = []

        x = zoom_xoffset
        y = zoom_yoffset

        for image in imagelist:
            # Append the bounding box
            input_tilemap.append([image[1], x, y, x + zoom_x_size, y + zoom_y_size])
            print "Laid out image %s at bounding box (%d, %d -> %d, %d)" % (image[1], x, y, x + zoom_x_size, y + zoom_y_size)
            direction = image[0]
            if direction == 'r':
                x += zoom_x_size
            if direction == 'd':
                y += zoom_y_size
                # Don't forget to "carriage return" after a down shift
                x = zoom_xoffset

        print "Generating output file tilemap"

        output_tilemap = []
        temp_x = 0
        temp_y = 0
        for tile in range(0, int(tiles_for_zoom)):
            print "Generating collisions for tile %d at (%d, %d -> %d, %d)" % (tile, temp_x, temp_y, temp_x+tilesize, temp_y+tilesize)
            tile = []
            for input_tile in input_tilemap:
                print "Attempting to collide boxes"
                box_x = temp_x if temp_x > input_tile[1] else input_tile[1]
                box_y = temp_y if temp_y > input_tile[2] else input_tile[2]
                box_x_final = temp_x + tilesize if temp_x + tilesize < input_tile[3] else input_tile[3]
                box_y_final = temp_y + tilesize if temp_y + tilesize < input_tile[4] else input_tile[4]
                # Detect if the collision yields a valid bounding box
                if(box_x >= 0 and box_y >= 0 and box_x_final > box_x and box_y_final > box_y):
                    print "Collision detected with input tile at %d x %d -> %d x %d" % (input_tile[1], input_tile[2], input_tile[3], input_tile[4])
                    print "Bounding box of collision area %d x %d -> %d x %d" % (box_x, box_y, box_x_final, box_y_final)

                    # Now translate these directly into relative coordinates for PIL
                    # Note we've kept everything as floats up to here, but Python makes rounding errors instead of doing the right thing (yay floats!)
                    # So instead of rounding to int at the end, we round each element to an int. This makes the calculations much smoother
                    input_start_x = int((box_x) - (input_tile[1]))
                    input_stop_x = int((box_x_final) - (input_tile[1]))
                    #if((temp_x == 0 or temp_y == 0) and box_x_final == temp_x + tilesize):
                    #    input_stop_x += 1
                    if(input_stop_x + 1 < (input_tile[3] - input_tile[1])):
                        input_stop_x += 1
                    input_start_y = int((box_y) - (input_tile[2]))
                    input_stop_y = int((box_y_final) - (input_tile[2]))
                    if(input_stop_y + 1 < (input_tile[4] - input_tile[2])):
                        input_stop_y += 1
                    output_start_x = int((box_x) - (temp_x))
                    output_start_y = int((box_y) - (temp_y))
                    
                    print "Will copy from %d x %d -> %d x %d" % (input_start_x, input_start_y, input_stop_x, input_stop_y)
                    print "Will paste to %d x %d" % (output_start_x, output_start_y)
                    tile.append([input_tile[0], input_start_x, input_start_y, input_stop_x, input_stop_y, output_start_x, output_start_y])

            output_tilemap.append([None, temp_x / tilesize, temp_y / tilesize, tile])
            # Increment the tile position
            if temp_x + tilesize == square_for_zoom:
                temp_x = 0
                temp_y += tilesize
            else:
                temp_x += tilesize

        print "Performing output tile generation"

        for image in imagelist:
            print "Loading image %s" % image[1]
            im = Image.open(image[1])
            if(crop):
                im = im.crop((crop[0], crop[1], crop[2], crop[3]))
            if(debug):
                quality = Image.NEAREST
            else:
                quality = Image.ANTIALIAS
            im = im.resize((int(zoom_x_size), int(zoom_y_size)), quality)
            print "Image mode %s" % im.mode
            # Loop over output tiles
            for output_tile in output_tilemap:
                # Loop over their pending operations
                for operation in output_tile[3]:
                    # If an operation matches our image
                    if operation and operation[0] == image[1]:
                        print "Found an operation matching image in tile %d x %d" % (output_tile[1], output_tile[2])
                        # If the image hasn't been created yet
                        if(output_tile[0] == None):
                            print "Output tile was missing. Created."
                            output_tile[0] = Image.new('RGB', (tilesize, tilesize), (255,)*4)
                        # Perform the tile transfer
                        output_tile[0].paste(im.crop((operation[1], operation[2], operation[3], operation[4])), (operation[5], operation[6]))
                        # Delete the operation we just did
                        print "Before screwing up the operations we had %s" % ','.join(map(str,output_tile[3]))
                        output_tile[3].remove(operation)
                        print "After we had %s" % ','.join(map(str,output_tile[3]))
                # If all operations have been completed
                if output_tile[0] and len(output_tile[3]) == 0:
                    print "Found a completed tile, saving tile %d x %d" % (output_tile[1], output_tile[2])
                    if not os.path.exists("%s/%d/%d" % (prefix, zoom, output_tile[1])):
                        os.makedirs("%s/%d/%d" % (prefix, zoom, output_tile[1]))
                    # If its a totally blank tile let's make it now to save ourselves
                    if(output_tile[0] == None):
                        print "Output tile was missing. Created."
                        output_tile[0] = Image.new('RGB', (tilesize, tilesize), (255,)*4)
                    output_tile[0].save("%s/%d/%d/%d.jpg" % (prefix, zoom, output_tile[1], output_tile[2]))
                    # Delete the image and stop us from trying to save it again
                    output_tile[0] = False

def write_settings(max, size, prefix):
    print "Writing map settings configuration..."
    output = {}
    output['max'] = max
    output['size'] = size
    json_output = json.dumps(output)
    if not os.path.exists("%s" % prefix):
        os.makedirs("%s" % prefix)
    output_file = open("%s/map_configuration.json" % prefix, 'w')
    output_file.write(json_output)
    output_file.close()

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description="Tile a set of input images for OpenLayers or Google Maps")
    parser.add_argument('--imagecfg', metavar='file', type=argparse.FileType('r'), dest='configfile', required=True, help='configuration file containing position of source images')
    parser.add_argument('--destination', metavar='path', dest='prefix', required=True, help='location to save generated tiles (will be created if it does not exist)')
    parser.add_argument('--draft', action='store_true', dest='debug', help='Lower resize quality for faster image production. Good for testing tiling')
    args = parser.parse_args()
    imagelist = []
    for line in args.configfile:
        split = line.split(',')
        imagelist.append((split[1].rstrip(), split[0]))
    #crop = (1206, 522, 11180, 14000)
    crop = False
    tile(256, imagelist, args.prefix, args.debug, crop)

if __name__ == "__main__":
    sys.exit(main())
