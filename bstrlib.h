/*
 * This source file is part of the bstring string library.  This code was
 * written by Paul Hsieh in 2002-2008, and is covered by the BSD open source 
 * license and the GPL. Refer to the accompanying documentation for details 
 * on usage and license.
 */

/*
 * bstrlib.c
 *
 * This file is the core module for implementing the bstring functions.
 */

#ifndef BSTRLIB_INCLUDE
#define BSTRLIB_INCLUDE

#ifdef __cplusplus
extern "C" {
#endif

#include "plm_config.h"

#include <stdarg.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>

#if !defined (BSTRLIB_VSNP_OK) && !defined (BSTRLIB_NOVSNP)
# if defined (__TURBOC__) && !defined (__BORLANDC__)
#  define BSTRLIB_NOVSNP
# endif
#endif

#define BSTR_ERR (-1)
#define BSTR_OK (0)
#define BSTR_BS_BUFF_LENGTH_GET (0)

typedef struct tagbstring * bstring;
typedef const struct tagbstring * const_bstring;

/* Copy functions */
#define cstr2bstr bfromcstr
gpuit_EXPORT bstring bfromcstr (const char * str);
gpuit_EXPORT bstring bfromcstralloc (int mlen, const char * str);
gpuit_EXPORT bstring blk2bstr (const void * blk, int len);
gpuit_EXPORT char * bstr2cstr (const_bstring s, char z);
gpuit_EXPORT int bcstrfree (char * s);
gpuit_EXPORT bstring bstrcpy (const_bstring b1);
gpuit_EXPORT int bassign (bstring a, const_bstring b);
gpuit_EXPORT int bassignmidstr (bstring a, const_bstring b, int left, int len);
gpuit_EXPORT int bassigncstr (bstring a, const char * str);
gpuit_EXPORT int bassignblk (bstring a, const void * s, int len);

/* Destroy function */
gpuit_EXPORT int bdestroy (bstring b);

/* Space allocation hinting functions */
gpuit_EXPORT int balloc (bstring s, int len);
gpuit_EXPORT int ballocmin (bstring b, int len);

/* Substring extraction */
gpuit_EXPORT bstring bmidstr (const_bstring b, int left, int len);

/* Various standard manipulations */
gpuit_EXPORT int bconcat (bstring b0, const_bstring b1);
gpuit_EXPORT int bconchar (bstring b0, char c);
gpuit_EXPORT int bcatcstr (bstring b, const char * s);
gpuit_EXPORT int bcatblk (bstring b, const void * s, int len);
gpuit_EXPORT int binsert (bstring s1, int pos, const_bstring s2, unsigned char fill);
gpuit_EXPORT int binsertch (bstring s1, int pos, int len, unsigned char fill);
gpuit_EXPORT int breplace (bstring b1, int pos, int len, const_bstring b2, unsigned char fill);
gpuit_EXPORT int bdelete (bstring s1, int pos, int len);
gpuit_EXPORT int bsetstr (bstring b0, int pos, const_bstring b1, unsigned char fill);
gpuit_EXPORT int btrunc (bstring b, int n);

/* Scan/search functions */
gpuit_EXPORT int bstricmp (const_bstring b0, const_bstring b1);
gpuit_EXPORT int bstrnicmp (const_bstring b0, const_bstring b1, int n);
gpuit_EXPORT int biseqcaseless (const_bstring b0, const_bstring b1);
gpuit_EXPORT int bisstemeqcaselessblk (const_bstring b0, const void * blk, int len);
gpuit_EXPORT int biseq (const_bstring b0, const_bstring b1);
gpuit_EXPORT int bisstemeqblk (const_bstring b0, const void * blk, int len);
gpuit_EXPORT int biseqcstr (const_bstring b, const char * s);
gpuit_EXPORT int biseqcstrcaseless (const_bstring b, const char * s);
gpuit_EXPORT int bstrcmp (const_bstring b0, const_bstring b1);
gpuit_EXPORT int bstrncmp (const_bstring b0, const_bstring b1, int n);
gpuit_EXPORT int binstr (const_bstring s1, int pos, const_bstring s2);
gpuit_EXPORT int binstrr (const_bstring s1, int pos, const_bstring s2);
gpuit_EXPORT int binstrcaseless (const_bstring s1, int pos, const_bstring s2);
gpuit_EXPORT int binstrrcaseless (const_bstring s1, int pos, const_bstring s2);
gpuit_EXPORT int bstrchrp (const_bstring b, int c, int pos);
gpuit_EXPORT int bstrrchrp (const_bstring b, int c, int pos);
#define bstrchr(b,c) bstrchrp ((b), (c), 0)
#define bstrrchr(b,c) bstrrchrp ((b), (c), blength(b)-1)
gpuit_EXPORT int binchr (const_bstring b0, int pos, const_bstring b1);
gpuit_EXPORT int binchrr (const_bstring b0, int pos, const_bstring b1);
gpuit_EXPORT int bninchr (const_bstring b0, int pos, const_bstring b1);
gpuit_EXPORT int bninchrr (const_bstring b0, int pos, const_bstring b1);
gpuit_EXPORT int bfindreplace (bstring b, const_bstring find, const_bstring repl, int pos);
gpuit_EXPORT int bfindreplacecaseless (bstring b, const_bstring find, const_bstring repl, int pos);

/* List of string container functions */
struct bstrList {
    int qty, mlen;
    bstring * entry;
};
gpuit_EXPORT struct bstrList * bstrListCreate (void);
gpuit_EXPORT int bstrListDestroy (struct bstrList * sl);
gpuit_EXPORT int bstrListAlloc (struct bstrList * sl, int msz);
gpuit_EXPORT int bstrListAllocMin (struct bstrList * sl, int msz);

/* String split and join functions */
gpuit_EXPORT struct bstrList * bsplit (const_bstring str, unsigned char splitChar);
gpuit_EXPORT struct bstrList * bsplits (const_bstring str, const_bstring splitStr);
gpuit_EXPORT struct bstrList * bsplitstr (const_bstring str, const_bstring splitStr);
gpuit_EXPORT bstring bjoin (const struct bstrList * bl, const_bstring sep);
gpuit_EXPORT int bsplitcb (const_bstring str, unsigned char splitChar, int pos,
	int (* cb) (void * parm, int ofs, int len), void * parm);
gpuit_EXPORT int bsplitscb (const_bstring str, const_bstring splitStr, int pos,
	int (* cb) (void * parm, int ofs, int len), void * parm);
gpuit_EXPORT int bsplitstrcb (const_bstring str, const_bstring splitStr, int pos,
	int (* cb) (void * parm, int ofs, int len), void * parm);

/* Miscellaneous functions */
gpuit_EXPORT int bpattern (bstring b, int len);
gpuit_EXPORT int btoupper (bstring b);
gpuit_EXPORT int btolower (bstring b);
gpuit_EXPORT int bltrimws (bstring b);
gpuit_EXPORT int brtrimws (bstring b);
gpuit_EXPORT int btrimws (bstring b);

#if !defined (BSTRLIB_NOVSNP)
gpuit_EXPORT bstring bformat (const char * fmt, ...);
gpuit_EXPORT int bformata (bstring b, const char * fmt, ...);
gpuit_EXPORT int bassignformat (bstring b, const char * fmt, ...);
gpuit_EXPORT int bvcformata (bstring b, int count, const char * fmt, va_list arglist);

#define bvformata(ret, b, fmt, lastarg) { \
bstring bstrtmp_b = (b); \
const char * bstrtmp_fmt = (fmt); \
int bstrtmp_r = BSTR_ERR, bstrtmp_sz = 16; \
	for (;;) { \
		va_list bstrtmp_arglist; \
		va_start (bstrtmp_arglist, lastarg); \
		bstrtmp_r = bvcformata (bstrtmp_b, bstrtmp_sz, bstrtmp_fmt, bstrtmp_arglist); \
		va_end (bstrtmp_arglist); \
		if (bstrtmp_r >= 0) { /* Everything went ok */ \
			bstrtmp_r = BSTR_OK; \
			break; \
		} else if (-bstrtmp_r <= bstrtmp_sz) { /* A real error? */ \
			bstrtmp_r = BSTR_ERR; \
			break; \
		} \
		bstrtmp_sz = -bstrtmp_r; /* Doubled or target size */ \
	} \
	ret = bstrtmp_r; \
}

#endif

typedef int (*bNgetc) (void *parm);
typedef size_t (* bNread) (void *buff, size_t elsize, size_t nelem, void *parm);

/* Input functions */
gpuit_EXPORT bstring bgets (bNgetc getcPtr, void * parm, char terminator);
gpuit_EXPORT bstring bread (bNread readPtr, void * parm);
gpuit_EXPORT int bgetsa (bstring b, bNgetc getcPtr, void * parm, char terminator);
gpuit_EXPORT int bassigngets (bstring b, bNgetc getcPtr, void * parm, char terminator);
gpuit_EXPORT int breada (bstring b, bNread readPtr, void * parm);

/* Stream functions */
gpuit_EXPORT struct bStream * bsopen (bNread readPtr, void * parm);
gpuit_EXPORT void * bsclose (struct bStream * s);
gpuit_EXPORT int bsbufflength (struct bStream * s, int sz);
gpuit_EXPORT int bsreadln (bstring b, struct bStream * s, char terminator);
gpuit_EXPORT int bsreadlns (bstring r, struct bStream * s, const_bstring term);
gpuit_EXPORT int bsread (bstring b, struct bStream * s, int n);
gpuit_EXPORT int bsreadlna (bstring b, struct bStream * s, char terminator);
gpuit_EXPORT int bsreadlnsa (bstring r, struct bStream * s, const_bstring term);
gpuit_EXPORT int bsreada (bstring b, struct bStream * s, int n);
gpuit_EXPORT int bsunread (struct bStream * s, const_bstring b);
gpuit_EXPORT int bspeek (bstring r, const struct bStream * s);
gpuit_EXPORT int bssplitscb (struct bStream * s, const_bstring splitStr, 
	int (* cb) (void * parm, int ofs, const_bstring entry), void * parm);
gpuit_EXPORT int bssplitstrcb (struct bStream * s, const_bstring splitStr, 
	int (* cb) (void * parm, int ofs, const_bstring entry), void * parm);
gpuit_EXPORT int bseof (const struct bStream * s);

struct tagbstring {
	int mlen;
	int slen;
	unsigned char * data;
};

/* Accessor macros */
#define blengthe(b, e)      (((b) == (void *)0 || (b)->slen < 0) ? (int)(e) : ((b)->slen))
#define blength(b)          (blengthe ((b), 0))
#define bdataofse(b, o, e)  (((b) == (void *)0 || (b)->data == (void*)0) ? (char *)(e) : ((char *)(b)->data) + (o))
#define bdataofs(b, o)      (bdataofse ((b), (o), (void *)0))
#define bdatae(b, e)        (bdataofse (b, 0, e))
#define bdata(b)            (bdataofs (b, 0))
#define bchare(b, p, e)     ((((unsigned)(p)) < (unsigned)blength(b)) ? ((b)->data[(p)]) : (e))
#define bchar(b, p)         bchare ((b), (p), '\0')

/* Static constant string initialization macro */
#define bsStaticMlen(q,m)   {(m), (int) sizeof(q)-1, (unsigned char *) ("" q "")}
#if defined(_MSC_VER)
# define bsStatic(q)        bsStaticMlen(q,-32)
#endif
#ifndef bsStatic
# define bsStatic(q)        bsStaticMlen(q,-__LINE__)
#endif

/* Static constant block parameter pair */
#define bsStaticBlkParms(q) ((void *)("" q "")), ((int) sizeof(q)-1)

/* Reference building macros */
#define cstr2tbstr btfromcstr
#define btfromcstr(t,s) {                                            \
    (t).data = (unsigned char *) (s);                                \
    (t).slen = ((t).data) ? ((int) (strlen) ((char *)(t).data)) : 0; \
    (t).mlen = -1;                                                   \
}
#define blk2tbstr(t,s,l) {            \
    (t).data = (unsigned char *) (s); \
    (t).slen = l;                     \
    (t).mlen = -1;                    \
}
#define btfromblk(t,s,l) blk2tbstr(t,s,l)
#define bmid2tbstr(t,b,p,l) {                                                \
    const_bstring bstrtmp_s = (b);                                           \
    if (bstrtmp_s && bstrtmp_s->data && bstrtmp_s->slen >= 0) {              \
        int bstrtmp_left = (p);                                              \
        int bstrtmp_len  = (l);                                              \
        if (bstrtmp_left < 0) {                                              \
            bstrtmp_len += bstrtmp_left;                                     \
            bstrtmp_left = 0;                                                \
        }                                                                    \
        if (bstrtmp_len > bstrtmp_s->slen - bstrtmp_left)                    \
            bstrtmp_len = bstrtmp_s->slen - bstrtmp_left;                    \
        if (bstrtmp_len <= 0) {                                              \
            (t).data = (unsigned char *)"";                                  \
            (t).slen = 0;                                                    \
        } else {                                                             \
            (t).data = bstrtmp_s->data + bstrtmp_left;                       \
            (t).slen = bstrtmp_len;                                          \
        }                                                                    \
    } else {                                                                 \
        (t).data = (unsigned char *)"";                                      \
        (t).slen = 0;                                                        \
    }                                                                        \
    (t).mlen = -__LINE__;                                                    \
}
#define btfromblkltrimws(t,s,l) {                                            \
    int bstrtmp_idx = 0, bstrtmp_len = (l);                                  \
    unsigned char * bstrtmp_s = (s);                                         \
    if (bstrtmp_s && bstrtmp_len >= 0) {                                     \
        for (; bstrtmp_idx < bstrtmp_len; bstrtmp_idx++) {                   \
            if (!isspace (bstrtmp_s[bstrtmp_idx])) break;                    \
        }                                                                    \
    }                                                                        \
    (t).data = bstrtmp_s + bstrtmp_idx;                                      \
    (t).slen = bstrtmp_len - bstrtmp_idx;                                    \
    (t).mlen = -__LINE__;                                                    \
}
#define btfromblkrtrimws(t,s,l) {                                            \
    int bstrtmp_len = (l) - 1;                                               \
    unsigned char * bstrtmp_s = (s);                                         \
    if (bstrtmp_s && bstrtmp_len >= 0) {                                     \
        for (; bstrtmp_len >= 0; bstrtmp_len--) {                            \
            if (!isspace (bstrtmp_s[bstrtmp_len])) break;                    \
        }                                                                    \
    }                                                                        \
    (t).data = bstrtmp_s;                                                    \
    (t).slen = bstrtmp_len + 1;                                              \
    (t).mlen = -__LINE__;                                                    \
}
#define btfromblktrimws(t,s,l) {                                             \
    int bstrtmp_idx = 0, bstrtmp_len = (l) - 1;                              \
    unsigned char * bstrtmp_s = (s);                                         \
    if (bstrtmp_s && bstrtmp_len >= 0) {                                     \
        for (; bstrtmp_idx <= bstrtmp_len; bstrtmp_idx++) {                  \
            if (!isspace (bstrtmp_s[bstrtmp_idx])) break;                    \
        }                                                                    \
        for (; bstrtmp_len >= bstrtmp_idx; bstrtmp_len--) {                  \
            if (!isspace (bstrtmp_s[bstrtmp_len])) break;                    \
        }                                                                    \
    }                                                                        \
    (t).data = bstrtmp_s + bstrtmp_idx;                                      \
    (t).slen = bstrtmp_len + 1 - bstrtmp_idx;                                \
    (t).mlen = -__LINE__;                                                    \
}

/* Write protection macros */
#define bwriteprotect(t)     { if ((t).mlen >=  0) (t).mlen = -1; }
#define bwriteallow(t)       { if ((t).mlen == -1) (t).mlen = (t).slen + ((t).slen == 0); }
#define biswriteprotected(t) ((t).mlen <= 0)

#ifdef __cplusplus
}
#endif

#endif
