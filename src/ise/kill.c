/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use Sergey's algorithm for killing processes. */
void
kill_igpax (void)
{
    HANDLE snapshot;
    HANDLE process;
    PROCESSENTRY32 pe;
    BOOL process_found;
    BOOL rc;

    snapshot = CreateToolhelp32Snapshot (TH32CS_SNAPPROCESS, 0);
    process_found = Process32First (snapshot, &pe);
    while (process_found) {
	//printf ("process: %s\n",pe.szExeFile);
	process_found = Process32Next (snapshot, &pe);
	if (!_stricmp(pe.szExeFile,"igpax.exe")) {
	    /* Strangely, I seem to have enough priveleges to kill the 
	       process, even without calling AdjustTokenPrivileges() */
	    process = OpenProcess (PROCESS_ALL_ACCESS, 0, pe.th32ProcessID);
	    if (!process) {
		printf ("Open failed\n");
		continue;
	    }
	    rc = TerminateProcess (process, 0);
	    if (!rc) {
		printf ("Kill failed\n");
		continue;
	    }
	    printf ("Kill succeeded\n");
	}
    }
    CloseHandle (snapshot);
}
