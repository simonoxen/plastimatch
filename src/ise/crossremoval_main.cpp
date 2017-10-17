#include "crossremoval.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	CrossRemoval w;
	w.show();
	return a.exec();
}
