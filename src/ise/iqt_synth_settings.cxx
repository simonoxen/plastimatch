/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QtGui>
#include <QTimer>
#include <QFileDialog>
#include <QDir>
#include "iqt_video_widget.h"
//#include <vtkPolyDataMapper.h>
//#include <vtkRenderer.h>
//#include <vtkRenderWindow.h>
//#include <vtkSphereSource.h>
//#include "vtkSmartPointer.h"

#include "iqt_application.h"
#include "iqt_main_window.h"
#include "iqt_synth_settings.h"

Iqt_synth_settings::Iqt_synth_settings (QWidget *parent)
    : QDialog (parent)
{
    /* Set up GUI */
    setupUi (this);
    this->mw = (Iqt_main_window*) parent;
}

Iqt_synth_settings::~Iqt_synth_settings ()
{
    QSettings settings;
    settings.sync ();    
}

void
Iqt_synth_settings::slot_cancel ()
{
    this->close();
}

void
Iqt_synth_settings::slot_proceed ()
{
/*#if defined (commentout)
    QMessageBox::information (0, QString ("Info"), 
	QString ("slot_synth() was called"));
#endif
    ise_app->set_synthetic_source ();*/
    rows = rowBox->value();
    cols = colBox->value();
    ampl = ampBox->value();
    mark = markBox->value();
    noise = noiseBox->value();
    //qDebug("Setting 1: %d", rows);
    //Iqt_main_window::slot_synth_set (set1/*, set2, set3, set4, set5*/);

    ise_app->set_synthetic_source (mw, rows, cols, ampl, mark, noise);
    this->close();
    
}

void
Iqt_synth_settings::slot_default ()
{
    rowBox->setValue(1536);
    colBox->setValue(2048);
    ampBox->setValue(1);
    markBox->setValue(0);
    noiseBox->setValue(0);
}


