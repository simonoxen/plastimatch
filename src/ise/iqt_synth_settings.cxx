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

Iqt_synth_settings::Iqt_synth_settings ()
{
    /* Set up GUI */
    setupUi (this);
}

#if defined (commentout)
void
Iqt_synth_settings::render_sphere ()
{
    // sphere
    vtkSmartPointer<vtkSphereSource> sphereSource = 
	vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->Update();
    vtkSmartPointer<vtkPolyDataMapper> sphereMapper =
	vtkSmartPointer<vtkPolyDataMapper>::New();
    sphereMapper->SetInputConnection(sphereSource->GetOutputPort());
    vtkSmartPointer<vtkActor> sphereActor = 
	vtkSmartPointer<vtkActor>::New();
    sphereActor->SetMapper(sphereMapper);
 
    // VTK Renderer
    vtkSmartPointer<vtkRenderer> renderer = 
	vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(sphereActor);
 
    // VTK/Qt wedded
    this->qvtkWidget->GetRenderWindow()->AddRenderer(renderer);
};
#endif

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
    set1 = spinBox->value();
    qDebug("Setting 1: %d", set1);
    //Iqt_main_window::slot_synth_set (set1/*, set2, set3, set4, set5*/);
    this->close();
    
}
/*
void
Iqt_synth_settings::slot_settings ();
{
    set1 = spinBox.value()
}
*/

