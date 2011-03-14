/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#ifndef __FRAME_H__
#define __FRAME_H__

void frame_clear (Frame* f);
void frame_autosense (Frame* f, int prev_dark, unsigned long rows, unsigned long cols);
int frame_is_dark (Frame* f);
int frame_is_bright (Frame* f);
int frame_needs_write (Frame* f);

#endif
