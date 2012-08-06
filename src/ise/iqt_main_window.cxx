/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Libraries */
#include <QtGui>
#include <QByteArray>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QTimer>
#include <QLabel>
#include <QMutex>
#include <QSpinBox>
#include <QWaitCondition>
#include <stdio.h>

/* Plastimatch headers */
#include "ise_config.h"
#include "cbuf.h"
#include "frame.h"
#include "his_io.h"
#include "iqt_synth_settings.h"
#include "iqt_application.h"
#include "iqt_main_window.h"
#include "iqt_tracker.h"
#include "iqt_video_widget.h"
#include "sleeper.h"
#include "synthetic_source_thread.h"
#include "tracker_thread.h"

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
    num_track->setHidden(true);
    track_label->setHidden(true);
    /* Start the timer */
    //    m_qtimer = new QTimer (this);
    //    connect (m_qtimer, SIGNAL(timeout()), this, SLOT(slot_timer()));
    //    m_qtimer->start(1000);
    connect (this, SIGNAL(fluoro_ready(QString)), this, SLOT(show_fluoro(QString)));
    this->playing = false;
    this->synth = false;
    framePos = new QSlider (Qt::Horizontal, this);
    connect (framePos, SIGNAL(valueChanged(int)), this, SLOT(get_new_frame(int)));
    slider_layout->addWidget(framePos);
    frameNum = 0;
    tracker = new Tracker (this);
}

Iqt_main_window::~Iqt_main_window ()
{
    QSettings settings;
    settings.sync ();
    delete framePos;
    //delete m_qtimer;
    delete tracker;
}

void
Iqt_main_window::slot_load ()
{
    if (playing) {
        this->slot_pause ();
	}
    if (synth){
        setMax->setRange(32500, 65000);
        setMin->setRange(32500, 65000);
        setMax->setValue(65000);
        setMin->setValue(63000);
    }
    playing = false;
    const QString DEFAULT_DIR_KEY(QDir::homePath());

    QSettings settings;
    QString ddk = settings.value(DEFAULT_DIR_KEY).toString();
    if (ddk.isEmpty()) {
        ddk = QDir::homePath();
    }
    filename = QFileDialog::getOpenFileName(this, "Select a file",
        ddk, tr("Image Files (*.his *.jpg *.png *.bmp)"));

    if (!filename.isEmpty()) {
	QDir CurrentDir;
        settings.setValue(DEFAULT_DIR_KEY, CurrentDir.absoluteFilePath(filename));
    }
    
    if (filename.isNull()) {
        return;
    }

    statusBar()->showMessage(QString("Filename: %1")
        .arg(filename));

    QString path = QFileInfo(filename).path();
    show_fluoro(path);
    vid_screen->rescale();
}

void
Iqt_main_window::show_fluoro (QString path)
{
    QDir directory = QDir(path);
    QStringList files = directory.entryList(QDir::Files, QDir::Name);
    numFiles = files.size();
    framePos->setMaximum(numFiles-1);

    for (int j=0; j < numFiles; j++) {
	filename = path + "/" + files.at(j);
	
	QByteArray ba = filename.toLocal8Bit();
	const char *fn = ba.data();
	
	if (j == 0 && is_his (512, 512, fn)) {
	    ise_app->cbuf[0]->clear();
	    ise_app->cbuf[0]->init (0, numFiles, 512, 512);
	}

        Frame *f = ise_app->cbuf[0]->get_frame ();
        bool isHis = his_read (f->img, 512, 512, fn);

        if (isHis) {
            ise_app->cbuf[0]->add_waiting_frame (f);
        } else {
            ise_app->cbuf[0]->add_empty_frame (f);
	}
	frameList[j] = f;
    }

    ise_app->cbuf[0]->display_lock_oldest_frame ();
    this->slot_frame_ready (512, 512);
    playing = false;
    this->slot_play ();
}

void
Iqt_main_window::slot_save ()
{
    QMessageBox::information (0, QString ("Info"), 
	QString ("slot_save() was called"));
}

void
Iqt_main_window::slot_play ()
{   
    if (playing) {
	return;
    } else {
	playing = true;
        play_pause_button->setText ("||");
        action_Play->setText ("&Pause");
	while (playing) {
	    ise_app->mutex.lock();
	    if (!isTracking) {
		framePos->setValue(frameNum);
		Sleeper::msleep(200);
	    } else {
		framePos->setValue(frameNum);
		qDebug() << "Displaying frame " << frameNum;
		Sleeper::msleep(200);
		ise_app->frameLoaded.wakeAll();
	    }
	    ise_app->mutex.unlock();
	    QCoreApplication::processEvents();
	    frameNum++;
	    if (frameNum==numFiles) {
		playing = false;
		if (isTracking) slot_set_tracking (false);
	    }
	}
    }
}

void
Iqt_main_window::slot_pause ()
{
    if (!playing) {
	return;
    } else {
	playing = false;
	play_pause_button->setText ("|>");
	action_Play->setText ("&Play");
    }
}

void
Iqt_main_window::get_new_frame (int pos)
{
    ise_app->cbuf[0]->display_lock_frame (pos);
    slot_reload_frame();
}

void
Iqt_main_window::slot_go_back ()
{
    framePos->setValue(0);
    frameNum = 0;
    playing = false;
}

void
Iqt_main_window::slot_go_forward ()
{
    framePos->setValue(numFiles);
    frameNum = numFiles;
    playing = false;
}

void
Iqt_main_window::slot_stop ()
{
    if (playing) {
	playing = false;
        play_pause_button->setText ("|>");
        action_Play->setText ("&Play");
	slot_go_back ();
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
    setMax->setRange(0, 1000);
    setMin->setRange(0, 1000);
    setMax->setValue(800);
    setMin->setValue(200);
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
    qDebug("Reloading frame...");
    if (setMax->isSliderDown() && setMax->value() <= setMin->value())
    {
	setMin->setValue(setMax->value());
    } else if (setMin->isSliderDown() && setMax->value() <= setMin->value())
    {
	setMax->setValue(setMin->value());
    }

    int max = setMax->value();
    int min = setMin->value();

    unsigned short min_val = 0xffff;
    unsigned short max_val = 0;

    Frame *f = *(ise_app->cbuf[0]->display_ptr);
    for (int i = 0; i < width * height; i++) {
        if (f->img[i] < min_val) min_val = f->img[i];
        if (f->img[i] > max_val) max_val = f->img[i];
    }
    
    
    uchar *imgdata = new uchar[width * height * 4];
    for (int i = 0; i < width * height; i++) {
        float fval = (f->img[i] - min) * 255.0 / (max-min);
        if (fval < 0) fval = 0; else if (fval > 255) fval = 255;
        uchar val = (uchar) fval;
        imgdata[4*i+0] = val;
        imgdata[4*i+1] = val;
        imgdata[4*i+2] = val;
        imgdata[4*i+3] = 0xff;
    }
    QImage qimage (imgdata, width, height, QImage::Format_RGB32);
    this->playing = true;

    vid_screen->set_qimage (qimage);
    delete imgdata;
}

void
Iqt_main_window::slot_frame_ready (int width, int height)
{
    qDebug("slot_frame_ready");
    if (setMin->isHidden())
    {
	setMin->setHidden(false);
	min_label->setHidden(false);
	min_val->setHidden(false);
	setMax->setHidden(false);
	max_label->setHidden(false);
	max_val->setHidden(false);
    }

    this->width = width;
    this->height = height;
    
    this->slot_reload_frame();
}

void
Iqt_main_window::slot_set_tracking (bool clicked)
{
    if (clicked){
	num_track->setHidden(false);
	track_label->setHidden(false);
	//this->tracker->tracker_initialize ();
	this->tracker->tracker_thread->start ();
	isTracking = true;
    } else {
	num_track->setHidden(true);
	track_label->setHidden(true);
	this->tracker->tracker_thread->quit ();
	isTracking = false;
    }
}

void
Iqt_main_window::trackPoint (double row, double col)
{
    this->vid_screen->pix = QPointF (row, col);
    this->vid_screen->manual = false;
    this->vid_screen->updateTracking ();
}
