#include <string>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>


namespace TCLAP {
    class MyOutput : public StdOutput
    {
    public:
	virtual void failure(CmdLineInterface& c, ArgException& e)
	{ 
	    std::cerr << "My special failure message for: " << std::endl
		      << e.what() << std::endl;
	    exit(1);
	}

	virtual void usage (CmdLineInterface& _cmd)
	{
	    std::ostream& os = std::cout;
	    std::list<Arg*> argList = _cmd.getArgList();
	    std::string message = _cmd.getMessage();
	    XorHandler xorHandler = _cmd.getXorHandler();
	    std::vector< std::vector<Arg*> > xorList = xorHandler.getXorList();

	    // first the xor 
	    for ( int i = 0; static_cast<unsigned int>(i) < xorList.size(); i++ )
	    {
		for ( ArgVectorIterator it = xorList[i].begin(); 
		      it != xorList[i].end(); 
		      it++ )
		{
		    spacePrint( os, (*it)->longID(), 75, 3, 3 );
		    spacePrint( os, (*it)->getDescription(), 75, 5, 0 );

		    if ( it+1 != xorList[i].end() )
			spacePrint(os, "-- OR --", 75, 9, 0);
		}
		os << std::endl << std::endl;
	    }

	    // then the rest
	    for (ArgListIterator it = argList.begin(); it != argList.end(); it++)
		if ( !xorHandler.contains( (*it) ) )
		{
		    std::string s = "   " + (*it)->longID();
		    int len = s.length();
		    if (len > 25) {
			os << s << std::endl;
			s = (*it)->getDescription();
			spacePrint (os, s, 75, 25, 27);
		    } else {
			for (int i = len; i < 25; i++) {
			    s += ' ';
			}
			s += (*it)->getDescription();
			spacePrint (os, s, 75, 0, 27);
		    }
		    //os << std::endl;
		}

	    os << std::endl;

	    spacePrint( os, message, 75, 3, 0 );
	}


	virtual void version(CmdLineInterface& c)
	{
	    std::cout << "my version message: 0.1" << std::endl;
	}

	inline void spacePrint( std::ostream& os, 
	    const std::string& s, 
	    int maxWidth, 
	    int indentSpaces, 
	    int secondLineOffset ) const
	{
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
#if defined (commentout)
#endif
    };
}

int main(int argc, char** argv)
{

    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {  

	// Define the command line object, and insert a message
	// that describes the program. The "Command description message" 
	// is printed last in the help text. The second argument is the 
	// delimiter (usually space) and the last one is the version number. 
	// The CmdLine object parses the argv array based on the Arg objects
	// that it contains. 
	TCLAP::CmdLine cmd("Command description message", ' ', "0.9");

	TCLAP::MyOutput my;
	cmd.setOutput (&my);
#if defined (commentout)
#endif

	// Define a value argument and add it to the command line.
	// A value arg defines a flag and a type of value that it expects,
	// such as "-n Bishop".
	TCLAP::ValueArg<std::string> nameArg("n","name","Name to print",true,"homer","string");

	TCLAP::ValueArg<std::string> randomArg("R","random","A random argument",false,"not-set","string");

	// Add the argument nameArg to the CmdLine object. The CmdLine object
	// uses this Arg to parse the command line.
	cmd.add( nameArg );
	cmd.add( randomArg );

	// Define a switch and add it to the command line.
	// A switch arg is a boolean argument and only defines a flag that
	// indicates true or false.  In this example the SwitchArg adds itself
	// to the CmdLine object as part of the constructor.  This eliminates
	// the need to call the cmd.add() method.  All args have support in
	// their constructors to add themselves directly to the CmdLine object.
	// It doesn't matter which idiom you choose, they accomplish the same thing.
	TCLAP::SwitchArg reverseSwitch ("r","reverse","Print name backwards", cmd, false);
	TCLAP::SwitchArg longSwitch ("q","qeverse",
	    "This is a not so brief description of the desired functionality. "
	    "Maybe it even spans multiple lines. "
	    "Maybe it even spans multiple lines. "
	    "Maybe it even spans multiple lines."
	    , cmd, false);
	TCLAP::SwitchArg anotherSwitch ("","qeverse-foobar","This is a not so brief description of the desired functionality", cmd, false);

	// Parse the argv array.
	cmd.parse( argc, argv );

	if (randomArg.isSet()) {
	    printf ("randomArg was set!\n");
	}
	printf ("randomArg = %s\n", randomArg.getValue().c_str());

	// Get the value parsed by each arg. 
	std::string name = nameArg.getValue();
	bool reverseName = reverseSwitch.getValue();

	// Do what you intend. 
	if ( reverseName )
	{
	    std::reverse(name.begin(),name.end());
	    std::cout << "My name (spelled backwards) is: " << name << std::endl;
	}
	else
	    std::cout << "My name is: " << name << std::endl;


    } catch (TCLAP::ArgException &e) { 
	std::cerr << "error: " << e.error() 
		  << " for arg " << e.argId() << std::endl; 
    }
    catch (...) {
	std::cerr << "???\n";
    }
}
