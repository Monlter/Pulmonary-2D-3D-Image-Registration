import SimpleITK as sitk
import argparse
import os
import numpy as np


def nii_trans_bin(path, spacing_list):
    ct_list = os.listdir(path)
    save_path = os.path.join(path, "..", "tran_bin")
    os.makedirs(save_path, exist_ok=True)
    for ct_name in ct_list:
        save_name = ct_name.split(".nii")[0] + ".bin"
        sitk_img = sitk.ReadImage(os.path.join(path, ct_name))
        sitk_img = ImageResample(sitk_img, spacing_list)
        np_img = sitk.GetArrayFromImage(sitk_img)
        shape = np_img.shape
        np_img.astype('float32').tofile(os.path.join(save_path, save_name))
        print("{} file have saved! shape:->{}".format(os.path.join(save_path, save_name), shape))


def ImageResample(sitk_image, spacing_list, is_label=False):
    size = np.array(sitk_image.GetSize())
    spacing = np.array(spacing_list)
    max_spacing_size = max(spacing)
    new_spacing = np.array([1, 1, 1])
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True, type=str)
    parser.add_argument('--spacing', '-s', required=True, type=float, nargs='+')
    args = parser.parse_args()
    nii_trans_bin(args.path, args.spacing)

