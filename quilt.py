#!/usr/bin/env python

"""

Quilt.

Image tiler for OpenLayers and Google Maps designed for large input sets

Copyright (C) 2012, 2017 Eric Harmon

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

from __future__ import print_function

import argparse
import json
import logging
import math
import os
import sys

from concurrent import futures
from PIL import Image
import yaml


# Disable decompression bomb warnings, we know we're working with large images
Image.MAX_IMAGE_PIXELS = 1000000000
logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class OutputTile(object):
    def __init__(self, zoom_level, tile_location, tile_size, prefix):
        self.zoom_level = zoom_level
        self.tile_location = tile_location
        self.tile_size = tile_size
        self.prefix = prefix

        self._operations = set()

        self._image = None

        self.tile_complete = False

    def __repr__(self):
        return "<tile %d/%d/%d>" % (self.zoom_level, self.tile_location[0], self.tile_location[1])

    def add_operation(self, operation):
        """
        Add an operation, this will automatically set the output tile as incomplete since it needs to be written again.
        """
        self.tile_complete = False

        self._operations.add(operation)

    def remove_operation(self, operation):
        """
        Remove an operation.
        """
        LOGGER.info("%s: Removed operation", self)
        self._operations.remove(operation)

    def operations_for_image(self, image):
        """
        Find all the operations matching a particular image.
        """
        return [operation for operation in self._operations if operation.input_image == image]

    def add_image_data(self, operation, image_data):
        """
        Given input image data, copy it into the tile.
        """
        LOGGER.info("%s: Adding image data for image '%s'", self, operation.input_image.name)
        if not self._image:
            LOGGER.info("%s: Created blank output tile", self)
            self._image = Image.new('RGBA', (self.tile_size, self.tile_size), (255, 0, 0, 0))

        # If image_data is None, we're being asked to create a blank tile, so just continue without copying anything
        if image_data is not None:
            LOGGER.info("%s: Copied data from image '%s' to output tile", self, operation.input_image.name)
            self._image.paste(image_data.crop(operation.input_location), operation.output_location)

        LOGGER.info("%s: Trying to remove operation", self)
        self.remove_operation(operation)
        LOGGER.info("%s: Operation removed", self)

    @property
    def operations_complete(self):
        """
        Check if we've completed all operations.
        """
        return not bool(self._operations)

    def finalize(self):
        """
        Finalize the tile by writing it to disk.
        """
        if not self.operations_complete:
            raise Exception("%s: Cannot finalize, operations incomplete." % self)

        if self._image:
            LOGGER.info("%s: Saving tile at zoom %s", self, self.zoom_level)
            self._image.save("%s/%d/%d/%d.png" % (self.prefix, self.zoom_level, self.tile_location[0], self.tile_location[1]))

        # Unset the image data so garbage collect can do it's thing
        self._image = None

        self.tile_complete = True

    def safe_finalize(self):
        """
        Try to finalize, but only if operations are complete.
        """
        LOGGER.info("%s: Was asked to try finalizing", self)
        if self.operations_complete:
            self.finalize()
        else:
            LOGGER.info("%s: Not ready to finalize", self)


class InputImage(object):
    def __init__(self, name, path=None):
        self.name = name
        self.path = path


class ImageOperation(object):
    def __init__(self, input_image, input_location, output_location):
        self.input_image = input_image
        self.input_location = input_location
        self.output_location = output_location



def tile(tilesize, config, prefix, debug, crop):
    """Generate a list of tiles given a list of images and metadata"""

    LOGGER.info("Generating %d sized tiles to %s from %d images", tilesize, prefix, len(config['images']))

    LOGGER.info("Determining overall landscape size...")

    # Let's calculate the size of our full image scape
    im = Image.open(config['images'][0]['file'])
    if(crop):
        im = im.crop((crop[0], crop[1], crop[2], crop[3]))
    size = im.size

    # Free the image for GC
    del im

    full_x_size = size[0]
    full_y_size = size[1]

    x_full_landscape = config['width'] * full_x_size
    y_full_landscape = config['height'] * full_y_size

    LOGGER.info("Total landscape size: %d x %d", x_full_landscape, y_full_landscape)

    square_full_landscape = x_full_landscape if x_full_landscape > y_full_landscape else y_full_landscape

    LOGGER.info("Square landscape size: %d x %d", square_full_landscape, square_full_landscape)

    LOGGER.info("Determining maximum zoom level...")

    # Essentially zoom levels are 2^(zoom level) tiles per side. This means we need to see what the closest power of 2 to the number of tiles we need is.
    estimated_tiles_per_side = square_full_landscape / tilesize
    # Reverse the power of 2 with a log
    max_zoom = int(math.ceil(math.log(estimated_tiles_per_side, 2)))

    LOGGER.info("Maximum zoom level is %d", max_zoom)

    LOGGER.info("Pregenerating source image tilemap...")

    LOGGER.info("Determining start offsets...")

    xoffset = (1.0 * square_full_landscape - x_full_landscape) / 2
    yoffset = (1.0 * square_full_landscape - y_full_landscape) / 2

    LOGGER.info("Offsets %d x %d", xoffset, yoffset)

    # Subtacting 1 from max_zoom and computing a scale factor to offset since we're disallowing maximum zoom currently
    # TODO: Need to figure out how to more properly feed this data to openLayers
    tiles_per_side_for_zoom = math.pow(2, max_zoom)
    tiles_for_zoom = tiles_per_side_for_zoom * tiles_per_side_for_zoom
    square_for_zoom = tiles_per_side_for_zoom * tilesize
    scale_factor = (square_for_zoom / square_full_landscape)

    write_settings(max_zoom - 1, square_full_landscape * scale_factor, prefix)

    images = []
    for image in config['images']:
        images.append(InputImage(image['name'], image.get('file')))

    output_tilemap = {}
    output_tiles = []

    # This should be max_zoom + 1, disabled temporarily since the images are too huge
    final_max_zoom = max_zoom

    for zoom in range(0, final_max_zoom):

        LOGGER.info("Generating data for zoom level %d", zoom)

        output_tilemap[zoom] = {}

        tiles_per_side_for_zoom = math.pow(2, zoom)
        tiles_for_zoom = tiles_per_side_for_zoom * tiles_per_side_for_zoom
        square_for_zoom = tiles_per_side_for_zoom * tilesize

        LOGGER.info("\tWill generate %d tile(s) at landscape size %d x %d", tiles_for_zoom, square_for_zoom, square_for_zoom)

        LOGGER.info("\tDetermining zoomed landscape size...")

        scale_factor = (square_for_zoom / square_full_landscape)

        zoom_landscape_size_x = scale_factor * x_full_landscape
        zoom_landscape_size_y = scale_factor * y_full_landscape

        LOGGER.info("\tZoomed landscape size: %d x %d", zoom_landscape_size_x, zoom_landscape_size_y)

        zoom_xoffset = scale_factor * xoffset
        zoom_yoffset = scale_factor * yoffset

        LOGGER.info("\tZoomed offsets: %d x %d", zoom_xoffset, zoom_yoffset)

        zoom_x_size = scale_factor * full_x_size
        zoom_y_size = scale_factor * full_y_size

        LOGGER.info("\tGenerating input file tilemap")

        input_tilemap = []

        x = zoom_xoffset
        y = zoom_yoffset

        current_x_count = 0
        for image in images:
            # Append the bounding box
            input_tilemap.append([image, x, y, x + zoom_x_size, y + zoom_y_size])
            LOGGER.debug("\t\tLaid out image '%s' at bounding box (%d, %d -> %d, %d)", image.name, x, y, x + zoom_x_size, y + zoom_y_size)
            x += zoom_x_size
            current_x_count += 1
            if current_x_count == config['width']:
                current_x_count = 0
                y += zoom_y_size
                x = zoom_xoffset

            #direction = image[0]
            #if direction == 'r':
            #    x += zoom_x_size
            #if direction == 'd':
            #    y += zoom_y_size
            #    # Don't forget to "carriage return" after a down shift
            #    x = zoom_xoffset

        LOGGER.info("\tGenerating output file tilemap")

        zoom_output_tilemap = []
        temp_x = 0
        temp_y = 0
        for tile in range(0, int(tiles_for_zoom)):
            # Create an output tile for these parameters
            output_tile = OutputTile(zoom, (temp_x / tilesize, temp_y / tilesize), 256, prefix)
            output_tiles.append(output_tile)

            LOGGER.debug("\t\tGenerating collisions for tile %d at (%d, %d -> %d, %d)", tile, temp_x, temp_y, temp_x + tilesize, temp_y + tilesize)
            tile = []
            for input_tile in input_tilemap:
                #print "Attempting to collide boxes"
                box_x = temp_x if temp_x > input_tile[1] else input_tile[1]
                box_y = temp_y if temp_y > input_tile[2] else input_tile[2]
                box_x_final = temp_x + tilesize if temp_x + tilesize < input_tile[3] else input_tile[3]
                box_y_final = temp_y + tilesize if temp_y + tilesize < input_tile[4] else input_tile[4]
                # Detect if the collision yields a valid bounding box
                if(box_x >= 0 and box_y >= 0 and box_x_final > box_x and box_y_final > box_y):
                    LOGGER.debug("\t\t\tCollision detected with input tile at %d x %d -> %d x %d", input_tile[1], input_tile[2], input_tile[3], input_tile[4])
                    LOGGER.debug("\t\t\tBounding box of collision area %d x %d -> %d x %d", box_x, box_y, box_x_final, box_y_final)

                    # Now translate these directly into relative coordinates for PIL
                    # Note we've kept everything as floats up to here, but Python makes rounding errors instead of doing the right thing (yay floats!)
                    # So instead of rounding to int at the end, we round each element to an int. This makes the calculations much smoother
                    # TODO: Should we be rounding here instead? Or have we already lost precision with earlier
                    # calculations?
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
                    
                    LOGGER.debug("\t\t\tWill copy from %d x %d -> %d x %d", input_start_x, input_start_y, input_stop_x, input_stop_y)
                    LOGGER.debug("\t\t\tWill paste to %d x %d", output_start_x, output_start_y)
                    #tile.append([input_tile[0], input_start_x, input_start_y, input_stop_x, input_stop_y, output_start_x, output_start_y])

                    # Add an operation for this output tile to copy the source data in
                    output_tile.add_operation(ImageOperation(
                        input_tile[0],
                        (input_start_x, input_start_y, input_stop_x, input_stop_y),
                        (output_start_x, output_start_y),
                    ))

            #output_tilemap.append([None, temp_x / tilesize, temp_y / tilesize, tile])
            # Increment the tile position
            if temp_x + tilesize == square_for_zoom:
                temp_x = 0
                temp_y += tilesize
            else:
                temp_x += tilesize

    LOGGER.info("\tPerforming output tile generation")

    for image in images:
        LOGGER.info("\t\tProcessing image '%s'", image.name)

        LOGGER.info("\t\t\tLoading image")

        # If we have a filename load it and resize it
        if image.path:
            LOGGER.info("\t\t\t\tfrom file '%s'", image.path)
            # Convert to RGB
            # TODO? Technically they could have a transparency layer, but cheating and using RGB saves 25% memory space.
            im = Image.open(image.path).convert('RGB')
            if(crop):
                im = im.crop((crop[0], crop[1], crop[2], crop[3]))
            if(debug):
                quality = Image.NEAREST
            else:
                quality = Image.ANTIALIAS

        # If we don't have a filename, create a blank image to cut pieces out of
        else:
            LOGGER.info("\t\t\t\tusing blank image")
            im = Image.new('RGBA', (full_x_size, full_y_size), (255, 0, 0, 0))
            quality = Image.NEAREST

        for zoom in range(0, final_max_zoom):
            # TODO: Stop doing this twice.
            tiles_per_side_for_zoom = math.pow(2, zoom)
            tiles_for_zoom = tiles_per_side_for_zoom * tiles_per_side_for_zoom
            square_for_zoom = tiles_per_side_for_zoom * tilesize

            LOGGER.info("\tWill generate %d tile(s) at landscape size %d x %d", tiles_for_zoom, square_for_zoom,
                        square_for_zoom)

            LOGGER.info("\tDetermining zoomed landscape size...")

            scale_factor = (square_for_zoom / square_full_landscape)

            zoom_landscape_size_x = scale_factor * x_full_landscape
            zoom_landscape_size_y = scale_factor * y_full_landscape

            LOGGER.info("\tZoomed landscape size: %d x %d", zoom_landscape_size_x, zoom_landscape_size_y)

            zoom_xoffset = scale_factor * xoffset
            zoom_yoffset = scale_factor * yoffset

            LOGGER.info("\tZoomed offsets: %d x %d", zoom_xoffset, zoom_yoffset)

            zoom_x_size = scale_factor * full_x_size
            zoom_y_size = scale_factor * full_y_size
            # TODO: End TODO

            LOGGER.info("\tResizing input image")
            small_im = im.resize((int(zoom_x_size), int(zoom_y_size)), quality)

            # print "\t\tImage mode %s" % im.mode
            # Loop over output tiles
            def process_tiles(output_tile):
                operations = output_tile.operations_for_image(image)
                if operations:
                    LOGGER.info("\t\t\tFound an operation matching image in %s", output_tile)
                    for operation in operations:
                        output_tile.add_image_data(operation, small_im)
                        output_tile.safe_finalize()
                else:
                    LOGGER.info("\t\t\tNo operations found for this image at this size.")

                # If the image hasn't been created yet
                #if (output_tile[0] == None):
                #    LOGGER.debug("\t\t\tOutput tile was missing. Created.")
                #    output_tile[0] = Image.new('RGBA', (tilesize, tilesize), (255, 0, 0, 0))
                # Perform the tile transfer
                #output_tile[0].paste(small_im.crop(*output_tile.input_location), *output_tile.output_location)
                #output_tile[0].paste(small_im.crop((operation[1], operation[2], operation[3], operation[4])),
                #                     (operation[5], operation[6]))
                # Delete the operation we just did
                # print "Before screwing up the operations we had %s" % ','.join(map(str, output_tile[3]))
                #output_tile[3].remove(operation)
                # print "After we had %s" % ','.join(map(str,output_tile[3]))
                # If all operations have been completed
                #if output_tile[0] and len(output_tile[3]) == 0:
                #    LOGGER.debug("\t\t\tFound a completed tile, saving tile %d x %d", output_tile[1], output_tile[2])
                #    if not os.path.exists("%s/%d/%d" % (prefix, zoom, output_tile[1])):
                #        os.makedirs("%s/%d/%d" % (prefix, zoom, output_tile[1]))
                #    # If its a totally blank tile let's make it now to save ourselves
                #   if (output_tile[0] == None):
                #        LOGGER.debug("\t\t\tOutput tile was missing. Created.")
                #        output_tile[0] = Image.new('RGBA', (tilesize, tilesize), (255, 0, 0, 0))
                #    output_tile[0].save("%s/%d/%d/%d.png" % (prefix, zoom, output_tile[1], output_tile[2]))
                #    # Delete the image and stop us from trying to save it again
                #    output_tile[0] = False

            LOGGER.info("\t\t\tProcessing tiles for zoom %s", zoom)
            # TODO: We're accidentally ignoring exceptions here
            with futures.ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(process_tiles, [output_tile for output_tile in output_tiles if output_tile.zoom_level == zoom])
            #map(process_tiles, [output_tile for output_tile in output_tiles if output_tile.zoom_level == zoom])

    # TODO: After this we need a pass to finalize all images so the completely blank ones get written

def write_settings(max, size, prefix):
    print("Writing map settings configuration...")
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
    parser.add_argument('--config', metavar='file', type=argparse.FileType('r'), dest='config_file', required=True, help='configuration file containing position of source images')
    parser.add_argument('--destination', metavar='path', dest='prefix', required=True, help='location to save generated tiles (will be created if it does not exist)')
    parser.add_argument('--draft', action='store_true', dest='debug', help='Lower resize quality for faster image production. Good for testing tiling')
    args = parser.parse_args()
    imagelist = []
    config = yaml.load(args.config_file)
    #crop = (1206, 522, 11180, 14000)
    crop = False
    tile(256, config, args.prefix, args.debug, crop)

if __name__ == "__main__":
    sys.exit(main())
