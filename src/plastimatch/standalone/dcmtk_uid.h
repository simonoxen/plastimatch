/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __dcmtk_uid_h__
#define __dcmtk_uid_h__

#ifdef __cplusplus
extern "C" {
#endif
char*
plm_generate_dicom_uid (char *uid, const char *uid_root);
#ifdef __cplusplus
}
#endif

#endif
