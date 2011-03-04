/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bstrlib.h"
#include "img_metadata.h"
#include "plm_int.h"

Img_metadata::Img_metadata ()
{
}

Img_metadata::~Img_metadata ()
{
}

uint32_t
Img_metadata::make_key (uint16_t key1, uint16_t key2)
{
    return (key1 << 16) | key2;
}

const char* 
Img_metadata::get_metadata (uint32_t key)
{
    return m_data[key];
}

const char* 
Img_metadata::get_metadata (uint16_t key1, uint16_t key2)
{
    return get_metadata (this->make_key (key1, key2));
}
