/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <stdio.h>
#include <QtGui>
#include <QTimer>
#include <QFileDialog>
#include <QDir>
#include <QByteArray>
//#include <vtkPolyDataMapper.h>
//#include <vtkRenderer.h>
//#include <vtkRenderWindow.h>
//#include <vtkSphereSource.h>
//#include "vtkSmartPointer.h"
#include "cbuf.h"
#include "frame.h"
#include "his_io.h"
#include "synthetic_source_thread.h"
#include "iqt_synth_settings.h"
#include "iqt_application.h"
#include "iqt_main_window.h"
#include "iqt_video_widget.h"

/* Some hints on displaying video.  It may be necessary to explore 
   multiple options.  Start with the QWidget option?

   Use a QGraphicsItem, which you put in a QGraphicsScene
   http://www.qtforum.org/article/35428/display-live-camera-video-in-a-graphics-item.html
   Use glTexture
   http://www.qtforum.org/article/20311/using-opengl-to-display-dib-bmp-format-from-video-for-windows.html
   Use a QWidget
   http://sourceforge.net/projects/qtv4lcapture/
   http://qt-apps.org/content/show.php/Qt+Opencv+webcam+viewer?content=89995
   http://doc.trolltech.com/qq/qq16-fader.html
   Use phonon
   http://doc.qt.nokia.com/latest/phonon-overview.html
*/

Iqt_main_window::Iqt_main_window ()
{
    /* Sets up the GUI */
    setupUi (this);
    
    /* Hides the sliders until the source is called */
    setMin->setHidden(true);
    min_label->setHidden(true);
    min_val->setHidden(true);
    setMax->setHidden(true);
    max_label->setHidden(true);
    max_val->setHidden(true);
    
    /* Start the timer */
    //    m_qtimer = new QTimer (this);
    //    connect (m_qtimer, SIGNAL(timeout()), this, SLOT(slot_timer()));
    //    m_qtimer->start(1000);
    
    this->playing = false;
    this->synth = false;

    /* Render a sphere ?? */
//    this->render_sphere ();
}

#if defined (commentout)
void
Iqt_main_window::render_sphere ()
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

Iqt_main_window::~Iqt_main_window ()
{
    QSettings settings;
    settings.sync ();

    //delete m_qtimer;
}

void
Iqt_main_window::slot_load ()
{
    if (playing) {
        this->slot_play_pause();
    }
    playing = false;
    const QString DEFAULT_DIR_KEY(QDir::homePath());

    QSettings settings;
    filename = QFileDialog::getOpenFileName(this, "Select a file",                                              settings.value(DEFAULT_DIR_KEY).toString(),                                       tr("Image Files (*.his *.jpg *.png *.bmp)"));

    if (!filename.isEmpty()) {
        QDir CurrentDir;
        settings.setValue(DEFAULT_DIR_KEY, CurrentDir.absoluteFilePath(filename));
    }
    
   
    //Iqt_video_widget::load();

    if (filename.isNull()) {
        return;
    }

    statusBar()->showMessage(QString("Filename: %1")
        .arg(filename));
    //    for (int j = 0x79b1; j < 0x7a47; j++) {
    //	filename = QFileInfo(filename).path() + "/0000" + hex << j;
    QByteArray ba = filename.toLocal8Bit();
    const char *fn = ba.data();

    if (is_his (512, 512, fn)) {
        ise_app->cbuf[0]->clear();
        ise_app->cbuf[0]->init (0, 2, 512, 512);

        Frame *f = ise_app->cbuf[0]->get_frame ();
        bool isHis = his_read (f->img, 512, 512, fn);
        if (isHis) {
            ise_app->cbuf[0]->add_waiting_frame (f);
            this->slot_frame_ready (f, 512, 512);
        } else {
            ise_app->cbuf[0]->add_empty_frame (f);
        }
    } else {
	vid_screen->load(filename);
    }
    // }
    //label->setText(QString("Filename: %1").arg(filename));
    // this->slot_play_pause();
}

void
Iqt_main_window::slot_load_fluoro ()
{
}

void
Iqt_main_window::slot_save ()
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("slot_save() was called"));
}

void
Iqt_main_window::slot_play_pause ()
{   
    if (playing) {
	playing = false;
        play_pause_button->setText ("|>");
        action_Play->setText ("&Play");
    } else {
	playing = true;
        play_pause_button->setText ("||");
        action_Play->setText ("&Pause");
    }
    vid_screen->play(playing);
}

void
Iqt_main_window::slot_go_back ()
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("slot_go_back() was called"));
}

void
Iqt_main_window::slot_go_forward ()
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("slot_go_forward() was called"));
}

void
Iqt_main_window::slot_stop ()
{
    if (playing) {
	playing = false;
        play_pause_button->setText ("|>");
        action_Play->setText ("&Play");
    } else {
		QMessageBox::information (0, QString ("Info"),
		 QString ("This video has already been stopped."));
	}
	vid_screen->stop();
	/* Checks if synthetic source is currently running
	   would segfault without this check */
	if (synth) {
	    ise_app->stop();
	}
}

void
Iqt_main_window::slot_synth ()
{
    Iqt_synth_settings iqt_synth_settings (this);
    iqt_synth_settings.exec();
}

void
Iqt_main_window::slot_timer ()
{
#if defined (commentout)
    this->video_widget;
    QMessageBox::information (0, QString ("Info"), 
	QString ("slot_timer() was called"));
#endif
    statusBar()->showMessage(QString("Num panels = %1")
        .arg(ise_app->num_panels));
}

void
Iqt_main_window::slot_reload_frame ()
{
    
    if (setMax->isSliderDown() && setMax->value() <= setMin->value())
    {
	setMin->setValue(setMax->value());
    } else if (setMin->isSliderDown() && setMax->value() <= setMin->value())
    {
	setMax->setValue(setMin->value());
    }

    int max = setMax->value(); //changed by sliders, alters bg darkness
    int min = setMin->value(); //changed by sliders, alters bg&rect darkness

    unsigned short min_val = 0xffff;
    unsigned short max_val = 0;
    for (int i = 0; i < width * height; i++) {
        if (f->img[i] < min_val) min_val = f->img[i];
        if (f->img[i] > max_val) max_val = f->img[i];
    }

    uchar *data = new uchar[width * height * 4];
    for (int i = 0; i < width * height; i++) {
        float fval = (f->img[i] - min) * 255.0 / (max-min);
        if (fval < 0) fval = 0; else if (fval > 255) fval = 255;
        uchar val = (uchar) fval;
        data[4*i+0] = val;  //bg red
        data[4*i+1] = val;  //bg green
        data[4*i+2] = val;  //bg blue
        data[4*i+3] = 0xff; //alpha
    }
    QImage qimage (data, width, height, QImage::Format_RGB32);
    this->playing = true;

    vid_screen->set_qimage (qimage);
}

void
Iqt_main_window::slot_frame_ready (Frame* f, int width, int height)
{
    qDebug("Got frame %p", f);
    
    setMin->setHidden(false);
    min_label->setHidden(false);
    min_val->setHidden(false);
    setMax->setHidden(false);
    max_label->setHidden(false);
    max_val->setHidden(false);
    
    this->f = f;
    this->width = width;
    this->height = height;
    
    this->slot_reload_frame();
}
