#include "dcmtk/config/osconfig.h"    /* make sure OS specific configuration is included first */
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/ofstd/ofstream.h"


int main(int /*argc*/, char * /*argv*/ [])
{
#if defined (commentout)
    char uid[100];
    cout << "Series Instance UID : " << dcmGenerateUniqueIdentifier(uid, SITE_SERIES_UID_ROOT) << endl;
    return( 0 );
#endif
}
