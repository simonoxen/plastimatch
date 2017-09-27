#include "badpixeldetector.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	BadPixelDetector w;
	w.show();
	return a.exec();
}
