/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Note: wxWidgets bug for RegisterHotKey ()
    http://lists.wxwidgets.org/pipermail/wx-users/2009-January/111237.html
*/

#include <wx/wx.h>
#include <wx/window.h>
#include <wx/filename.h>
#include "sqlite3.h"

void initialize_sqlite ();

class MyFrame: public wxFrame
{
public:

    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

    virtual bool OnInit ();
    void OnMenuQuit (wxCommandEvent& event);
    void OnMenuAbout (wxCommandEvent& event);
    void OnButtonOK (wxCommandEvent& event);
    void OnButtonCancel (wxCommandEvent& event);
    void OnHotKey1 (wxKeyEvent& event);
    void OnHotKey2 (wxKeyEvent& event);

    wxBitmap bitmap;
    wxTextCtrl *editbox_patient_name;
    wxTextCtrl *editbox_patient_id;

    DECLARE_EVENT_TABLE()
};

class MyApp: public wxApp
{
public:
    virtual bool OnInit();
};

enum
{
    ID_MENU_QUIT = 1,
    ID_MENU_ABOUT,
    ID_BUTTON_OK,
    ID_BUTTON_CANCEL,
    ID_EDIT_PATIENT_NAME,
    ID_EDIT_PATIENT_ID
};

BEGIN_EVENT_TABLE(MyFrame, wxFrame)
    EVT_MENU(ID_MENU_QUIT, MyFrame::OnMenuQuit)
    EVT_MENU(ID_MENU_ABOUT, MyFrame::OnMenuAbout)
    EVT_HOTKEY(0xB000, MyFrame::OnHotKey1)
    EVT_HOTKEY(0xB001, MyFrame::OnHotKey2)
    EVT_BUTTON(ID_BUTTON_OK, MyFrame::OnButtonOK)
    EVT_BUTTON(ID_BUTTON_CANCEL, MyFrame::OnButtonCancel)
END_EVENT_TABLE()

IMPLEMENT_APP(MyApp)

bool MyApp::OnInit()
{
    /* Initialize JPEG library */
    ::wxInitAllImageHandlers ();

    /* Create and initialize main window */
    MyFrame* frame = new MyFrame( wxT("Hello World"), wxPoint(50,50), wxSize(450,340) );
    frame->OnInit ();
    SetTopWindow (frame);

    /* Load recently used patient name/id values */
    // initialize_sqlite ();

    return TRUE;
}

MyFrame::MyFrame (const wxString& title, const wxPoint& pos, const wxSize& size)
    : wxFrame((wxFrame *)NULL, -1, title, pos, size)
{
    wxMenuBar *menuBar = new wxMenuBar;
    wxMenu *menuFile = new wxMenu;

    menuFile->Append (ID_MENU_ABOUT, wxT("&About..."));
    menuFile->AppendSeparator ();
    menuFile->Append (ID_MENU_QUIT, wxT("E&xit"));
    menuBar->Append (menuFile, wxT("&File"));
    this->SetMenuBar (menuBar);

    this->CreateStatusBar ();
    this->SetStatusText (wxT("Welcome to wxWindows!"));

    wxPanel *panel = new wxPanel (this, -1);

    wxButton *ok = new wxButton (panel, ID_BUTTON_OK, wxT("Ok"));
    wxButton *cancel = new wxButton (panel, ID_BUTTON_CANCEL, wxT("Cancel"));

    wxBoxSizer *vbox = new wxBoxSizer (wxVERTICAL);
//    wxBoxSizer *hbox1 = new wxBoxSizer (wxHORIZONTAL);
//    wxBoxSizer *hbox2 = new wxBoxSizer (wxHORIZONTAL);
    wxBoxSizer *hbox3 = new wxBoxSizer (wxHORIZONTAL);
    wxFlexGridSizer *fgs = new wxFlexGridSizer(2, 2, 9, 25);

    wxStaticText *label_patient_name =  new wxStaticText (panel, wxID_ANY, wxT("Patient Name"));
    wxStaticText *label_patient_id =  new wxStaticText (panel, wxID_ANY, wxT("Patient ID"));
    this->editbox_patient_name = new wxTextCtrl (panel, ID_EDIT_PATIENT_NAME);
    this->editbox_patient_id = new wxTextCtrl (panel, ID_EDIT_PATIENT_ID);

//    hbox1->Add(label1, 0, wxRIGHT, 8);
//    hbox1->Add(patient_name, 1);

//    hbox2->Add(label2, 0, wxRIGHT, 8);
//    hbox2->Add(patient_id, 1);

    fgs->Add (label_patient_name);
    fgs->Add (editbox_patient_name, 1, wxEXPAND);
    fgs->Add (label_patient_id);
    fgs->Add (editbox_patient_id, 1, wxEXPAND);
    fgs->AddGrowableCol (1, 1);

    hbox3->Add (ok);
    hbox3->AddSpacer (20);
    hbox3->Add (cancel);

//    vbox->Add (hbox1, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
//    vbox->Add (hbox2, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
    vbox->Add (fgs, 1, wxALL | wxEXPAND, 15);
    vbox->Add (hbox3, 0, wxALIGN_RIGHT | wxRIGHT | wxBOTTOM, 10);
    panel->SetSizer (vbox);
}

void MyFrame::OnMenuQuit (wxCommandEvent& WXUNUSED(event))
{
    this->Close (TRUE);
}

void MyFrame::OnButtonCancel (wxCommandEvent& WXUNUSED(event))
{
    this->Show (FALSE);
}

void MyFrame::OnButtonOK (wxCommandEvent& WXUNUSED(event))
{
    wxString patient_name, patient_id;

    /* Save a copy */
    this->bitmap.SaveFile (wxT("C:/tmp/tmp.jpg"), wxBITMAP_TYPE_JPEG);

    /* Validate input fields */
    patient_name = this->editbox_patient_name->GetValue ();
    if (patient_name.IsEmpty ()) {
	wxMessageBox (wxT("Please enter a patient name"),
	    wxT("MONDOSHOT"), wxOK | wxICON_INFORMATION, this);
	return;
    }
    patient_id = this->editbox_patient_id->GetValue ();
    if (patient_id.IsEmpty ()) {
	wxMessageBox (wxT("Please enter a patient id"),
	    wxT("MONDOSHOT"), wxOK | wxICON_INFORMATION, this);
	return;
    }

    /* Bundle up and send dicom */
    


}

void MyFrame::OnMenuAbout (wxCommandEvent& WXUNUSED(event))
{
    wxMessageBox (wxT("This is Mondoshot!"),
        wxT("MONDOSHOT"), wxOK | wxICON_INFORMATION, this);
}

bool MyFrame::OnInit ()
{
    bool rc;
    rc = this->RegisterHotKey (0xB000, 0, wxCharCodeWXToMSW(WXK_F11));
    rc = this->RegisterHotKey (0xB001, 0, wxCharCodeWXToMSW(WXK_F12));

    wxSize screenSize = wxGetDisplaySize();
    this->bitmap.Create (screenSize.x, screenSize.y);

    return true;
}

void MyFrame::OnHotKey1 (wxKeyEvent& WXUNUSED(event))
{
    /* Save screenshot */
    wxSize screenSize = wxGetDisplaySize();
    wxScreenDC dc;
    wxMemoryDC memDC;
    memDC.SelectObject (this->bitmap);
    memDC.Blit (0, 0, screenSize.x, screenSize.y, &dc, 0, 0);
    memDC.SelectObject (wxNullBitmap);

#if defined (commentout)
    /* This is how you would save to a bmp file */
    /* Why don't I have PNG support??? */
    wxString fname = wxFileName::CreateTempFileName (wxT("screenshot")) + ".bmp";
    wxMessageBox (fname, wxT("MONDOSHOT"), wxOK | wxICON_INFORMATION, this);
    bitmap.SaveFile (fname, wxBITMAP_TYPE_BMP);
#endif


    this->Show (TRUE);
}

void MyFrame::OnHotKey2 (wxKeyEvent& WXUNUSED(event))
{
    this->Close (TRUE);
}

void
initialize_sqlite ()
{
    int rc;
    sqlite3 *db;
    char *sql;
    char *sqlite3_err;

    rc = sqlite3_open ("C:/tmp/mondoshot.sqlite", &db);
    if (rc) {
	fprintf (stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
	sqlite3_close (db);
	exit (1);
    }

    sql = "CREATE TABLE IF NOT EXISTS foo ( oi INTEGER PRIMARY KEY, patient_name TEXT, patient_id TEXT, last_used DATE );";
    rc = sqlite3_exec (db, sql, 0, 0, &sqlite3_err);
    if (rc != SQLITE_OK) {
	fprintf (stderr, "SQL error: %s\n", sqlite3_err);
	sqlite3_free (sqlite3_err);
    }

    //sqlite> insert into foo (patient_name, patient_id, last_used) values ('foo', 'bar', julianday('now'));

    sqlite3_close (db);
}
