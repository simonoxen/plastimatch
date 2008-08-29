@rem perl dcm_image_uids.pl y:\database-dicom\0028\t5 0028_t5_uids.txt
@rem perl dcm_contour_points.pl y:\database-dicom\0028\103\1.2.840.113619.2.55.1.1762853477.1996.1155908038.536.103_0103_000513_11594884760bd1-no-phi.v2 0028_103_dump.cxt
@rem perl dcm_for_contour_propagation.pl 0028_103_dump.cxt 0028_t5_uids.txt 0028_merged.txt
@rem cxt_to_mha 0028_merged.txt 0028
@rem plastimatch 0028_parms.txt

@rem warp_mha --input=0028_esophagus.mha --output=0028_esophagus_warped.mha --vf=0028_vf.mha --interpolation nn
@rem warp_mha --input=0028_gtv-n1.mha --output=0028_gtv-n1_warped.mha --vf=0028_vf.mha --interpolation nn
@rem warp_mha --input=0028_gtv.mha --output=0028_gtv_warped.mha --vf=0028_vf.mha --interpolation nn
@rem warp_mha --input=0028_gtv_st8.mha --output=0028_gtv_st8_warped.mha --vf=0028_vf.mha --interpolation nn
@rem warp_mha --input=0028_heart.mha --output=0028_heart_warped.mha --vf=0028_vf.mha --interpolation nn
@rem warp_mha --input=0028_left_lung.mha --output=0028_left_lung_warped.mha --vf=0028_vf.mha --interpolation nn
@rem warp_mha --input=0028_right_lung.mha --output=0028_right_lung_warped.mha --vf=0028_vf.mha --interpolation nn
@rem warp_mha --input=0028_spinal_cord.mha --output=0028_spinal_cord_warped.mha --vf=0028_vf.mha --interpolation nn

@rem extract_contour 0028_esophagus_warped.mha 0028_esophagus_warped.txt
@rem extract_contour 0028_gtv-n1_warped.mha 0028_gtv-n1_warped.txt
@rem extract_contour 0028_gtv_warped.mha 0028_gtv_warped.txt
@rem extract_contour 0028_gtv_st8_warped.mha 0028_gtv_st8_warped.txt
@rem extract_contour 0028_heart_warped.mha 0028_heart_warped.txt
@rem extract_contour 0028_left_lung_warped.mha 0028_left_lung_warped.txt
@rem extract_contour 0028_right_lung_warped.mha 0028_right_lung_warped.txt
@rem extract_contour 0028_spinal_cord_warped.mha 0028_spinal_cord_warped.txt

perl make_dicomrt.pl 0028_t0_uids.txt 0028_esophagus_warped.txt 0028_gtv-n1_warped.txt 0028_gtv_warped.txt 0028_gtv_st8_warped.txt 0028_heart_warped.txt 0028_left_lung_warped.txt 0028_right_lung_warped.txt 0028_spinal_cord_warped.txt

@rem -------------------------------------------------------------
@rem dcmdump +L y:\database-dicom\0028\103\1.2.840.113619.2.55.1.1762853477.1996.1155908038.536.103_0103_000513_11594884760bd1-no-phi.v2 > 0028_103_dump.txt
@rem perl dcm_build_103.pl g:\reality\new-data\0028\103\1.2.840.113619.2.55.1.1762853477.1996.1155908038.536.103_0103_000019_11594884750bc9-no-phi.v2 g:\reality\0028_dump.txt 0028.txt 0028_prop.dcm
@rem perl dcm_build_103.pl g:\reality\new-data\0028\103\1.2.840.113619.2.55.1.1762853477.1996.1155908038.536.103_0103_000019_11594884750bc9-no-phi.v2 0028_dump_zw_2.txt 0028.txt 0028_prop.dcm
@rem dcmdump +L y:\database-dicom\0028\103\1.2.840.113619.2.55.1.1762853477.1996.1155908038.536.103_0103_000513_11594884760bd1-no-phi.v2 > 0028_103_dump.txt
