/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_tclap_h_
#define _plm_tclap_h_
#include "plm_config.h"
#include <string>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>

namespace TCLAP {

/* We need a replacement for longID, because it does not format well.  
   However, it is tedious to subclass all of the various subclasses of Arg.  
   Therefore we simply use a helper function. */
template<class T>
std::string long_id (T arg, const std::string& valueId = "val")
{
    std::string id = "";

    if (arg->getFlag() != "") {
	id += arg->flagStartString() + arg->getFlag();
	id += ", ";
    } else {
	id += "    ";
    }

    id += arg->nameStartString() + arg->getName();

    if (arg->isValueRequired()) {
	id += std::string( 1, arg->delimiter() ) + "<" + valueId + ">";
    }
    return id;
}

/* We need a replacement for the usage strings, because they do not 
   format well by default. */
class MyOutput : public StdOutput
{
public:
    virtual void failure(CmdLineInterface& c, ArgException& e) {
	std::cerr << "ERROR, " << e.error() << std::endl;
	brief_usage (c);
	usage (c);
	exit (1);
    }

    virtual void brief_usage (CmdLineInterface& cmd) {
	std::string prog_name = cmd.getProgramName();
	printf ("Usage: %s [options]\n", prog_name.c_str());
    }

    template<class T>
    void print_single_option (
	CmdLineInterface& _cmd, 
	std::ostream& os, 
	T arg
    )
    {
	std::string s = "   " + TCLAP::long_id(arg);
	int len = s.length();
	if (len > 25) {
	    os << s << std::endl;
	    s = arg->getDescription();
	    spacePrint (os, s, 75, 25, 27);
	} else {
	    for (int i = len; i < 25; i++) {
		s += ' ';
	    }
	    s += arg->getDescription();
	    spacePrint (os, s, 75, 0, 27);
	}
    }

    virtual void usage (CmdLineInterface& _cmd)	{
	std::ostream& os = std::cout;
	std::list<Arg*> argList = _cmd.getArgList();
	XorHandler xorHandler = _cmd.getXorHandler();
	std::vector< std::vector<Arg*> > xorList = xorHandler.getXorList();

	printf ("Options:\n");

	// first the xor 
	for (int i = 0; static_cast<unsigned int>(i) < xorList.size(); i++)
	{
	    for (ArgVectorIterator it = xorList[i].begin(); 
		 it != xorList[i].end(); 
		 it++)
	    {
		this->print_single_option (_cmd, os, *it);

		if ( it+1 != xorList[i].end() )
		    spacePrint(os, "-- OR --", 75, 9, 0);
	    }
	    os << std::endl << std::endl;
	}

	// then the rest
	for (ArgListIterator it = argList.begin(); it != argList.end(); it++)
	{
	    if (!xorHandler.contains ((*it)))
	    {
		this->print_single_option (_cmd, os, *it);
	    }
	}

	std::string message = 
	    "For documentation, see "
	    "<http://plastimatch.org>\n"
	    "For forum support, see "
	    "<http://groups.google.com/group/plastimatch>\n";
	os << message;
    }

    virtual void version(CmdLineInterface& c) {
	std::cout << "my version message: 0.1" << std::endl;
    }

#if defined (commentout)
    inline void spacePrint (
	std::ostream& os, 
	const std::string& s, 
	int maxWidth, 
	int indentSpaces, 
	int secondLineOffset) const {

	int len = static_cast<int>(s.length());
	if ( (len + indentSpaces > maxWidth) && maxWidth > 0 )
	{
	    int allowedLen = maxWidth - indentSpaces;
	    int start = 0;
	    while ( start < len )
	    {
		// find the substring length
		// int stringLen = std::min<int>( len - start, allowedLen );
		// doing it this way to support a VisualC++ 2005 bug 
		using namespace std; 
		int stringLen = min<int>( len - start, allowedLen );

		// trim the length so it doesn't end in middle of a word
		if ( stringLen == allowedLen )
		    while ( stringLen >= 0 &&
			s[stringLen+start] != ' ' && 
			s[stringLen+start] != ',' &&
			s[stringLen+start] != '|' ) 
			stringLen--;
	
		// ok, the word is longer than the line, so just split 
		// wherever the line ends
		if ( stringLen <= 0 )
		    stringLen = allowedLen;

		// check for newlines
		for ( int i = 0; i < stringLen; i++ )
		    if ( s[start+i] == '\n' )
			stringLen = i+1;

		// print the indent	
		for ( int i = 0; i < indentSpaces; i++ )
		    os << " ";

		if ( start == 0 )
		{
		    // handle second line offsets
		    indentSpaces += secondLineOffset;

		    // adjust allowed len
		    allowedLen -= secondLineOffset;
		}

		os << s.substr(start,stringLen) << std::endl;

		// so we don't start a line with a space
		while ( s[stringLen+start] == ' ' && start < len )
		    start++;
			
		start += stringLen;
	    }
	}
	else
	{
	    for ( int i = 0; i < indentSpaces; i++ )
		os << " ";
	    os << s << std::endl;
	}
    }
#endif
};

/* Another thing we need is proper sorting of the arguments. */
bool
option_sort (Arg*& lhs, Arg*& rhs)
{
    std::string lhs_string = lhs->getName();
    if (lhs->getFlag() != "") {
	lhs_string = lhs->getFlag();
    }
    std::string rhs_string = rhs->getName();
    if (rhs->getFlag() != "") {
	rhs_string = rhs->getFlag();
    }

    /* Put "--, --ignore_rest" last */
    if (lhs_string == "-") {
	return false;
    }
    if (rhs_string == "-") {
	return true;
    }

    /* This is tricky.  We need to sort -R after -r, etc. */
    for (unsigned int i = 0;
	 i<lhs_string.length() && i<rhs_string.length();
	 ++i) 
    {
	if (tolower(lhs_string[i]) < tolower(rhs_string[i])) {
	    return true;
	}
	if (tolower(lhs_string[i]) > tolower(rhs_string[i])) {
	    return false;
	}
    }
    if (lhs_string.length() < rhs_string.length()) {
	return true;
    } else {
	return false;
    }

    //return lhs_string < rhs_string;
}

void
sort_arglist (CmdLineInterface& cmd)
{
    std::list<Arg*>& arg_list = cmd.getArgList();

    arg_list.sort (option_sort);
}

} /* end namespace TCLAP */

#endif
