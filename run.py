# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2019. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import cv2
import logging
import numpy as np
from tempfile import TemporaryDirectory
from cytomine import CytomineJob
from cytomine.models import ImageInstance, ImageInstanceCollection, AnnotationCollection, Annotation
from cytomine.utilities.software import parse_domain_list
from shapely.geometry import Polygon
from sldc.locator import mask_to_objects_2d
from shapely.affinity import affine_transform
from skimage.filters import threshold_otsu

__author__ = "WSHMunirah WAhmad <wshmunirah@gmail.com>"

def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        
        images = ImageInstanceCollection()
        if cj.parameters.cytomine_id_images is not None:
            id_images = parse_domain_list(cj.parameters.cytomine_id_images)
            images.extend([ImageInstance().fetch(_id) for _id in id_images])
        else:
            images = images.fetch_with_filter("project", cj.parameters.cytomine_id_project)
        
        for image in cj.monitor(images, prefix="Running detection on image", period=0.1):
            # Resize image if needed
            resize_ratio = max(image.width, image.height) / cj.parameters.max_image_size
            if resize_ratio < 1:
                resize_ratio = 1

            bit_depth = image.bitDepth if image.bitDepth is not None else 8

            roi_annotations = AnnotationCollection()
            roi_annotations.terms=[cj.parameters.cytomine_id_roi_term]
            roi_annotations.project=cj.parameters.cytomine_id_project
            roi_annotations.image=image.id
            roi_annotations.showWKT = True
            if cj.parameters.cytomine_id_user:
              roi_annotations.user=cj.parameters.cytomine_id_user
            roi_annotations.fetch()
            print(roi_annotations)

            for roi in roi_annotations:
              roi_geometry = wkt.loads(roi.location)
              min_x=roi_geometry.bounds[0]
              min_y=roi_geometry.bounds[1]
              max_x=roi_geometry.bounds[2]
              max_y=roi_geometry.bounds[3]

              resized_width = int((max_x-min_x) / resize_ratio)
              resized_height = int((max_y-min_y) / resize_ratio)
              print(resized_width, resized_height)
          
              # download file in a temporary directory for auto-removal
              with TemporaryDirectory() as tmpdir:
                  download_path = os.path.join(tmpdir, "{id}.png")
                  roi.dump(dest_pattern=download_path, max_size=max(resized_width, resized_height),mask=True, alpha=True)
                  img_4ch = cv2.imread(download_path,cv2.IMREAD_UNCHANGED)
                
              ### to address alpha channel ###      
              img_alpha = img_4ch[:,:,3]
              img_rgb = img_4ch[:,:,:3]
              img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
              img = img_gray
              img[img_alpha==0]=255
              #################################

            pixels = np.array(img).flatten()
            th_value = threshold_otsu(pixels)
            print("Otsu threshold: ", th_value)
            threshold = th_value + cj.parameters.threshold_allowance
            print("Otsu threshold + allowance: ", threshold)
            thresh_mask = (img < threshold).astype(np.uint8)*255
          
            kernel_size = np.array(cj.parameters.kernel_size)
            if kernel_size.size != 2:  # noqa: PLR2004
              kernel_size = kernel_size.repeat(2)
            kernel_size = tuple(np.round(kernel_size).astype(int))
          
            # Create structuring element for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            min_region_size = np.sum(kernel)
            _, output, stats, _ = cv2.connectedComponentsWithStats(thresh_mask, connectivity=8)
            sizes = stats[1:, -1]
            for i, size in enumerate(sizes):
                if size < min_region_size:
                    thresh_mask[output == i + 1] = 0

            thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_DILATE, kernel)
            thresh_mask = cv2.bitwise_not(thresh_mask)
 
            extension = 10
            extended_img = cv2.copyMakeBorder(
                thresh_mask,
                extension,
                extension,
                extension,
                extension,
                cv2.BORDER_CONSTANT,
                value=2 ** bit_depth
            )

            # extract foreground polygons 
            fg_objects = mask_to_objects_2d(extended_img, background=255, offset=(-extension, -extension))
            zoom_factor = resize_ratio

            # Only keep components greater than {image_area_perc_threshold}% of whole image
            min_area = int((cj.parameters.image_area_perc_threshold / 100) * image.width * image.height)
            transform_matrix = [zoom_factor, 0, 0, -zoom_factor, min_x, max_y]
            annotations = AnnotationCollection()
            for i, (fg_poly, _) in enumerate(fg_objects):
                upscaled = affine_transform(fg_poly, transform_matrix)
                if upscaled.area <= min_area:
                    continue
                # print(upscaled.area)
                try:
                    print("Mask area: ", upscaled.area)
                    Annotation(
                    location=upscaled.wkt,
                    id_image=image.id,
                    id_terms=[cj.parameters.cytomine_id_predicted_term],
                    id_project=cj.parameters.cytomine_id_project).save()                    
                except:
                    print("An exception occurred. Proceed with next annotations")

        cj.job.update(statusComment="Finished.")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
