/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLPlastimatchLoadableNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMRMLPlastimatchLoadableNode.h"
#include "vtkMRMLScene.h"


//------------------------------------------------------------------------------
vtkMRMLPlastimatchLoadableNode* vtkMRMLPlastimatchLoadableNode::New()
{
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance (
	"vtkMRMLPlastimatchLoadableNode");
    if (ret) {
	return (vtkMRMLPlastimatchLoadableNode*) ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkMRMLPlastimatchLoadableNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLPlastimatchLoadableNode::CreateNodeInstance()
{
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance (
	"vtkMRMLPlastimatchLoadableNode");
    if (ret) {
	return (vtkMRMLPlastimatchLoadableNode*) ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkMRMLPlastimatchLoadableNode;
}

//----------------------------------------------------------------------------
vtkMRMLPlastimatchLoadableNode::vtkMRMLPlastimatchLoadableNode()
{
    this->Conductance = 1.0;
    this->NumberOfIterations = 1;
    this->TimeStep = 0.1;
    this->OutputVolumeRef = NULL;
    this->FixedVolumeRef = NULL;
    this->MovingVolumeRef = NULL;
    this->HideFromEditors = true;
}

//----------------------------------------------------------------------------
vtkMRMLPlastimatchLoadableNode::~vtkMRMLPlastimatchLoadableNode()
{
    this->SetOutputVolumeRef (NULL);
    this->SetFixedVolumeRef (NULL);
    this->SetMovingVolumeRef (NULL);
}

//----------------------------------------------------------------------------
void vtkMRMLPlastimatchLoadableNode::WriteXML(ostream& of, int nIndent)
{
    Superclass::WriteXML (of, nIndent);

    // Write all MRML node attributes into output stream
    vtkIndent indent(nIndent);
    {
	std::stringstream ss;
	ss << this->Conductance;
	of << indent << " Conductance=\"" << ss.str() << "\"";
    }
    {
	std::stringstream ss;
	ss << this->NumberOfIterations;
	of << indent << " NumberOfIterations=\"" << ss.str() << "\"";
    }
    {
	std::stringstream ss;
	ss << this->TimeStep;
	of << indent << " TimeStep=\"" << ss.str() << "\"";
    }
    if (this->OutputVolumeRef) {
	std::stringstream ss;
	ss << this->OutputVolumeRef;
	of << indent << " OutputVolumeRef=\"" << ss.str() << "\"";
    }
    if (this->FixedVolumeRef) {
	std::stringstream ss;
	ss << this->FixedVolumeRef;
	of << indent << " FixedVolumeRef=\"" << ss.str() << "\"";
    }
    if (this->MovingVolumeRef) {
	std::stringstream ss;
	ss << this->MovingVolumeRef;
	of << indent << " MovingVolumeRef=\"" << ss.str() << "\"";
    }
}

//----------------------------------------------------------------------------
void vtkMRMLPlastimatchLoadableNode::ReadXMLAttributes(const char** atts)
{
    vtkMRMLNode::ReadXMLAttributes(atts);

    // Read all MRML node attributes from two arrays of names and values
    const char* attName;
    const char* attValue;
    while (*atts != NULL) 
    {
	attName = *(atts++);
	attValue = *(atts++);
	if (!strcmp(attName, "Conductance")) 
	{
	    std::stringstream ss;
	    ss << attValue;
	    ss >> this->Conductance;
	}
	else if (!strcmp(attName, "NumberOfIterations")) 
	{
	    std::stringstream ss;
	    ss << attValue;
	    ss >> this->NumberOfIterations;
	}
	else if (!strcmp(attName, "TimeStep")) 
	{
	    std::stringstream ss;
	    ss << attValue;
	    ss >> this->TimeStep;
	}
	else if (!strcmp(attName, "OutputVolumeRef"))
	{
	    this->SetOutputVolumeRef(attValue);
	    this->Scene->AddReferencedNodeID(this->OutputVolumeRef, this);
	}
	else if (!strcmp(attName, "FixedVolumeRef"))
	{
	    this->SetFixedVolumeRef(attValue);
	    this->Scene->AddReferencedNodeID(this->FixedVolumeRef, this);
	}
	else if (!strcmp(attName, "MovingVolumeRef"))
	{
	    this->SetMovingVolumeRef(attValue);
	    this->Scene->AddReferencedNodeID(this->MovingVolumeRef, this);
	}
    }
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLPlastimatchLoadableNode::Copy(vtkMRMLNode *anode)
{
    Superclass::Copy(anode);
    vtkMRMLPlastimatchLoadableNode *node 
	= (vtkMRMLPlastimatchLoadableNode *) anode;
    this->SetConductance(node->Conductance);
    this->SetNumberOfIterations(node->NumberOfIterations);
    this->SetTimeStep(node->TimeStep);
    this->SetOutputVolumeRef(node->OutputVolumeRef);
    this->SetFixedVolumeRef(node->FixedVolumeRef);
    this->SetMovingVolumeRef(node->MovingVolumeRef);
}

//----------------------------------------------------------------------------
void vtkMRMLPlastimatchLoadableNode::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkMRMLNode::PrintSelf(os,indent);

    os << indent << "Conductance:   " << this->Conductance << "\n";
    os << indent << "NumberOfIterations:   " 
       << this->NumberOfIterations << "\n";
    os << indent << "TimeStep:   " << this->TimeStep << "\n";
    os << indent << "OutputVolumeRef:   " 
       << (this->OutputVolumeRef ? this->OutputVolumeRef : "(none)") << "\n";
    os << indent << "FixedVolumeRef:   " 
       << (this->FixedVolumeRef ? this->FixedVolumeRef : "(none)") << "\n";
    os << indent << "MovingVolumeRef:   " 
       << (this->MovingVolumeRef ? this->MovingVolumeRef : "(none)") << "\n";
}

//----------------------------------------------------------------------------
void vtkMRMLPlastimatchLoadableNode::UpdateReferenceID(const char *oldID, const char *newID)
{
    if (!strcmp(oldID, this->OutputVolumeRef))
    {
	this->SetOutputVolumeRef(newID);
    }
    if (!strcmp(oldID, this->FixedVolumeRef))
    {
	this->SetFixedVolumeRef(newID);
    }
    if (!strcmp(oldID, this->MovingVolumeRef))
    {
	this->SetMovingVolumeRef(newID);
    }
}
