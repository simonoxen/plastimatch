/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <stdio.h>
#include <stdlib.h>

#include "mabs_subject.h"
#include "volume.h"

Mabs_subject::Mabs_subject ()
{
    this->img_fn[0] = '\0';
    this->ss_fn[0] = '\0';
    this->img = NULL;
    this->ss = NULL;
    this->next = NULL;
}

Mabs_subject::~Mabs_subject ()
{
    delete this->img;
    delete this->ss;
}

Mabs_subject_manager::Mabs_subject_manager ()
{
    this->head = NULL;
    this->sel = NULL;
}

Mabs_subject_manager::~Mabs_subject_manager ()
{
    this->remove_all ();
}

Mabs_subject*
Mabs_subject_manager::add ()
{
    Mabs_subject* s = new Mabs_subject;
    s->next = this->head;
    this->head = s;

    return s;
}

bool
Mabs_subject_manager::remove (Mabs_subject* s)
{
    Mabs_subject* c = this->head;
    Mabs_subject* p = NULL;

    while (c != s) {
        if (!c) return false;
        p = c;
        c = c->next;
    }
    p->next = c->next;    
    delete c;

    if (this->sel == s) {
        this->sel = this->head;
    }

    return true;
}

void
Mabs_subject_manager::remove_all ()
{
    Mabs_subject* c = this->head;
    Mabs_subject* p = NULL;

    if (!this->head) return;

    while (c) {
        p = c;
        c = c->next;
        delete p;
    }
    this->head = NULL;
    this->sel = NULL;
}

void
Mabs_subject_manager::select_head ()
{
    this->sel = this->head;
}

bool
Mabs_subject_manager::select (Mabs_subject* s)
{
    this->sel = this->head;
    while (this->sel != s) {
        if (!this->sel) return false;
        this->sel = this->sel->next;
    }
    return true;
}

Mabs_subject*
Mabs_subject_manager::next ()
{
    this->sel = this->sel->next;
    return this->sel;
}

Mabs_subject*
Mabs_subject_manager::current ()
{
    return this->sel;
}
