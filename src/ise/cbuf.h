/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#ifndef __CBUF_H__
#define __CBUF_H__

#include <stdio.h>
#include "frame.h"

void cbuf_init (IseFramework* ig);
int cbuf_init_queue (IseFramework* ig, unsigned int idx, unsigned int num_frames);
void cbuf_append_waiting (CBuf* cbuf, Frame* new_frame);
void cbuf_insert_waiting (CBuf* cbuf, Frame* new_frame);
void cbuf_append_empty (CBuf* cbuf, Frame* new_frame);
Frame* cbuf_get_empty_frame (CBuf* cbuf);
Frame* cbuf_get_any_frame (CBuf* cbuf, int flush_dark);
unsigned long cbuf_queuelen (CBuf* cbuf, FrameQueue* queue);
int cbuf_shutdown (IseFramework* ig);
void cbuf_shutdown_queue (IseFramework* ig, CBuf* cbuf);
Frame* cbuf_display_lock_newest_frame (CBuf* cbuf);
Frame* cbuf_display_lock_newest_bright (CBuf* cbuf);
Frame* cbuf_display_lock_oldest_frame (CBuf* cbuf);
Frame* cbuf_display_lock_frame_by_idx (CBuf* cbuf, int frame_no);
Frame* cbuf_display_lock_next_frame (CBuf* cbuf);
Frame* cbuf_get_next_writable_frame (CBuf* cbuf);
void cbuf_reset_queue (CBuf* cbuf);
void cbuf_display_lock_release (CBuf* cbuf);
void cbuf_internal_grab_rewind (CBuf* cbuf);
Frame* cbuf_internal_grab_next_frame (CBuf* cbuf);
Frame* cbuf_display_lock_internal_grab (CBuf* cbuf);
void cbuf_append_waiting (CBuf* cbuf, Frame* new_frame);
void cbuf_mark_frame_written (CBuf* cbuf, Frame* frame);

#endif
