#!/usr/bin/env python

"""

Quilt.

Image tiler for OpenLayers and Google Maps designed for large input sets

Copyright (C) 2012, 2017-2018 Eric Harmon

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

from __future__ import division, print_function

import argparse
import json
import logging
import math
import os
import sys
import warnings

from concurrent import futures
from PIL import Image
import yaml


# Disable decompression bomb warnings, we know we're working with large images
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class OutputTile(object):
    """
    Manage an individual output tile.
    """
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
        LOGGER.debug("%s: Removed operation", self)
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
        LOGGER.debug("%s: Adding image data for image '%s'", self, operation.input_image.name)
        if not self._image:
            LOGGER.debug("%s: Created blank output tile", self)
            self._image = Image.new('RGBA', (self.tile_size, self.tile_size), (255, 0, 0, 0))

        # If image_data is None, we're being asked to create a blank tile, so just continue without copying anything
        if image_data is not None:
            LOGGER.debug("%s: Copied data from image '%s' to output tile", self, operation.input_image.name)
            self._image.paste(image_data.crop(operation.input_location), operation.output_location)

        LOGGER.debug("%s: Trying to remove operation", self)
        self.remove_operation(operation)
        LOGGER.debug("%s: Operation removed", self)

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
            LOGGER.debug("%s: Saving tile", self)
            location = "%s/%d/%d/%d.png" % (self.prefix, self.zoom_level, self.tile_location[0], self.tile_location[1])
            if not os.path.exists(os.path.dirname(location)):
                try:
                    os.makedirs(os.path.dirname(location))
                except OSError as err:
                    # Ignore directory creation collision when running in parralel
                    if err.errno != 17:
                        raise
            self._image.save(location)

        # Unset the image data so garbage collect can do it's thing
        self._image = None

        self.tile_complete = True

    def safe_finalize(self):
        """
        Try to finalize, but only if operations are complete.
        """
        LOGGER.debug("%s: Was asked to try finalizing", self)
        if self.operations_complete:
            self.finalize()
        else:
            LOGGER.debug("%s: Not ready to finalize", self)


class InputImage(object):
    """
    Manage an individual input image.
    """
    def __init__(self, name, path=None):
        self.name = name
        self.path = path


class ImageOperation(object):
    """
    Represent a single copy operation.
    """
    def __init__(self, input_image, input_location, output_location):
        self.input_image = input_image
        self.input_location = input_location
        self.output_location = output_location


class OutputData(object):
    def __init__(self, input_dimensions, input_image_dimensions, tile_size):
        self.input_dimensions = input_dimensions
        self.input_image_dimensions = input_image_dimensions
        self.tile_size = tile_size

    @property
    def raw_size(self):
        """
        Get the raw (non-square) size of the input image.
        """
        width = self.input_dimensions[0] * self.input_image_dimensions[0]
        height = self.input_dimensions[1] * self.input_image_dimensions[1]
        return width, height

    @property
    def size(self):
        """
        Get the (square) size of the output landscape.
        """
        width, height = self.raw_size
        if width > height:
            return width
        else:
            return height

    @property
    def maximum_zoom(self):
        """
        Determine the maximum zoom resolution we have based on the input image size.
        """
        # Essentially zoom levels are 2^(zoom level) tiles per side. This means we need to see what the closest power of
        # 2 to the number of tiles we need is.
        estimated_tiles_per_side = self.size / self.tile_size
        # Reverse the power of 2 with a log
        return int(math.ceil(math.log(estimated_tiles_per_side, 2)))

    @property
    def input_offsets(self):
        """
        Determine the placement offsets of the input image in the square output landscape.
        """
        x_offset = int(round((1.0 * self.size - self.raw_size[0]) / 2))
        y_offset = int(round((1.0 * self.size - self.raw_size[1]) / 2))

        return x_offset, y_offset

    def zoom(self, zoom):
        """
        Retrieve a zoom object representing the output data for a particular zoom level.
        """
        return ZoomData(self, zoom)


class ZoomData(object):
    """
    Data for a particular zoom level.
    """
    def __init__(self, output_data, zoom):
        self.output_data = output_data
        self.zoom = zoom

    @property
    def tiles_per_side(self):
        """
        Number of tiles on each side.
        """
        return int(math.pow(2, self.zoom))

    @property
    def tiles(self):
        """
        Total number of tiles.
        """
        return int(math.pow(self.tiles_per_side, 2))

    @property
    def size(self):
        """
        Size of the zoom in pixels.
        """
        return self.tiles_per_side * self.output_data.tile_size

    @property
    def scale_factor(self):
        """
        Scale factor relative to the full size of the image.
        """
        return self.size / self.output_data.size

    @property
    def input_offsets(self):
        """
        Offsets needed to center the image at this zoom level.
        """
        x_offset, y_offset = self.output_data.input_offsets
        return self.scale_factor * x_offset, self.scale_factor * y_offset

    @property
    def input_size(self):
        """
        Size of an individual input image after scaling to this zoom level.
        """
        x = int(round(self.scale_factor * self.output_data.input_image_dimensions[0]))
        y = int(round(self.scale_factor * self.output_data.input_image_dimensions[1]))
        return x, y


def tile(tile_size, config, prefix, debug, crop):
    """
    Generate a list of tiles given a list of images and metadata.
    """

    LOGGER.info("Generating %d sized tiles to %s from %d images", tile_size, prefix, len(config['images']))

    LOGGER.info("Determining overall landscape size...")
    # Let's calculate the size of our full image scape
    # TODO: Can't just select the first image, maybe it's blank
    im = Image.open(config['images'][0]['file'])
    if crop:
        im = im.crop(crop)
    size = im.size

    # Free the image for GC
    del im

    output_data = OutputData(
        (config['width'], config['height']),
        size,
        tile_size,
    )

    canvas_size = output_data.zoom(output_data.maximum_zoom).tiles_per_side * output_data.tile_size
    write_settings(output_data.maximum_zoom - 1, canvas_size, prefix)

    LOGGER.info("Building input image data...")
    images = []
    for image in config['images']:
        images.append(InputImage(image['name'], image.get('file')))

    output_tiles = []

    LOGGER.info("Calculating operations...")
    for zoom in range(0, output_data.maximum_zoom):
        LOGGER.info("\tGenerating data for zoom level %d...", zoom)
        zoom_data = output_data.zoom(zoom)

        input_tilemap = []

        x_offset, y_offset = zoom_data.input_offsets
        x, y = x_offset, y_offset
        zoom_x_size, zoom_y_size = zoom_data.input_size

        LOGGER.info("\t\tGenerating bounding boxes for input image data...")
        current_x_count = 0
        for image in images:
            # Append the bounding box
            input_tilemap.append([image, x, y, x + zoom_x_size, y + zoom_y_size])
            LOGGER.debug("\t\t\tLaid out image '%s' at bounding box (%d, %d -> %d, %d)", image.name, x, y, x + zoom_x_size, y + zoom_y_size)

            # Increment our tile position, resetting when we reach the right edge
            x += zoom_x_size
            current_x_count += 1
            if current_x_count == config['width']:
                current_x_count = 0
                y += zoom_y_size
                x = x_offset

        LOGGER.info("\t\tColliding bounding boxes with output tiles to generate operations...")

        tile_x = 0
        tile_y = 0
        for tile in range(0, int(zoom_data.tiles)):
            # Create an output tile for these parameters
            output_tile = OutputTile(zoom, (int(tile_x / tile_size), int(tile_y / tile_size)), 256, prefix)
            output_tiles.append(output_tile)

            LOGGER.debug("\t\tGenerating collisions for tile %d at (%d, %d -> %d, %d)", tile, tile_x, tile_y, tile_x + tile_size, tile_y + tile_size)
            for input_tile in input_tilemap:
                #print "Attempting to collide boxes"
                box_x = tile_x if tile_x > input_tile[1] else input_tile[1]
                box_y = tile_y if tile_y > input_tile[2] else input_tile[2]
                box_x_final = tile_x + tile_size if tile_x + tile_size < input_tile[3] else input_tile[3]
                box_y_final = tile_y + tile_size if tile_y + tile_size < input_tile[4] else input_tile[4]
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
                    output_start_x = int((box_x) - (tile_x))
                    output_start_y = int((box_y) - (tile_y))
                    
                    LOGGER.debug("\t\t\tWill copy from %d x %d -> %d x %d", input_start_x, input_start_y, input_stop_x, input_stop_y)
                    LOGGER.debug("\t\t\tWill paste to %d x %d", output_start_x, output_start_y)

                    # Add an operation for this output tile to copy the source data in
                    output_tile.add_operation(ImageOperation(
                        input_tile[0],
                        (input_start_x, input_start_y, input_stop_x, input_stop_y),
                        (output_start_x, output_start_y),
                    ))

            # Increment the tile position
            if tile_x + tile_size == zoom_data.size:
                tile_x = 0
                tile_y += tile_size
            else:
                tile_x += tile_size

    LOGGER.info("Performing output tile generation...")

    for image in images:
        LOGGER.info("\tProcessing image '%s'...", image.name)

        LOGGER.info("\t\tLoading image")

        # Initialize to nothing, throwing away results from the previous loop to save memory
        im = None

        # If we have a filename load it and resize it
        if image.path:
            LOGGER.info("\t\t\tfrom file '%s'.", image.path)
            im = Image.open(image.path)
            # If it's not already an RGB-based mode, convert it for full quality resizing
            if not im.mode.startswith('RGB'):
                im = im.convert('RGB')
            if crop:
                im = im.crop(crop)

            # If we're in draft mode, resize cheap and fast to save time
            if debug:
                quality = Image.NEAREST
            else:
                quality = Image.LANCZOS

        # If we don't have a filename, create a blank image to cut pieces out of
        else:
            LOGGER.info("\t\t\tusing blank image.")
            im = Image.new('RGBA', output_data.input_image_dimensions, (255, 0, 0, 0))
            # We can do a cheap resize since the pixels are already uniform
            quality = Image.NEAREST

        for zoom in range(0, output_data.maximum_zoom):
            LOGGER.info("\t\tProcessing zoom level %d", zoom)
            zoom_data = output_data.zoom(zoom)

            LOGGER.info("\t\t\tResizing input image...")
            small_im = im.resize(zoom_data.input_size, quality)

            # Loop over output tiles
            def process_tiles(output_tile):
                tiles = 0
                operations = output_tile.operations_for_image(image)
                if operations:
                    LOGGER.debug("\t\t\t\tFound an operation matching image in %s...", output_tile)
                    for operation in operations:
                        tiles += 1
                        output_tile.add_image_data(operation, small_im)
                        output_tile.safe_finalize()
                else:
                    LOGGER.debug("\t\tNo operations found for this image at this size.")

                return tiles

            LOGGER.info("\t\t\tProcessing tile operations...")

            # TODO: We're accidentally ignoring exceptions here
            with futures.ThreadPoolExecutor(max_workers=10) as executor:
                tiles = executor.map(process_tiles, [output_tile for output_tile in output_tiles if output_tile.zoom_level == zoom])
            LOGGER.info("\t\t\t%d tile operations processed.", sum(tiles))
            #map(process_tiles, [output_tile for output_tile in output_tiles if output_tile.zoom_level == zoom])
            # Clean up the resized copy early so we don't have it temporarily loaded twice when we come around the loop
            small_im = None

        LOGGER.info("\tDone with image '%s'.", image.name)

    # TODO: After this we need a pass to finalize all images so the completely blank ones get written

    LOGGER.info("Done processing tiles.")


def write_settings(max, size, prefix):
    print("Writing map settings configuration...")
    output = {
        'max': max,
        'size': size,
    }
    json_output = json.dumps(output)
    if not os.path.exists("%s" % prefix):
        os.makedirs("%s" % prefix)
    output_file = open("%s/map_configuration.json" % prefix, 'w')
    output_file.write(json_output)
    output_file.close()


def main():
    parser = argparse.ArgumentParser(description="Tile a set of input images for OpenLayers or Google Maps")
    parser.add_argument('--config', metavar='file', type=argparse.FileType('r'), dest='config_file', required=True, help='configuration file containing position of source images')
    parser.add_argument('--destination', metavar='path', dest='prefix', required=True, help='location to save generated tiles (will be created if it does not exist)')
    parser.add_argument('--draft', action='store_true', dest='debug', help='Lower resize quality for faster image production. Good for testing tiling')
    #parser.add_argument('--parallel', metavar='images', help="Number of input images to process in parallel. Greatly increases memory usage for large input images.")
    args = parser.parse_args()
    config = yaml.load(args.config_file)
    #crop = (1206, 522, 11180, 14000)
    crop = False
    tile(256, config, args.prefix, args.debug, crop)


if __name__ == "__main__":
    sys.exit(main())
