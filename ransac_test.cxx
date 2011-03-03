/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include "RANSAC.h"
#include "PlaneParametersEstimator.h"
#include "RandomNumberGenerator.h"
#include "autolabel_ransac_est.h"
#include "bstring_util.h"
#include "itk_point.h"
#include "print_and_exit.h"

const unsigned int DIMENSION = 2;
typedef std::vector< DoublePoint2DType > Point_vector;

void
load_data (
    Point_vector& data,
    const char* filename
)
{
    FILE *fp;
    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }
    CBStream bs ((bNread) fread, fp);

    CBString line;
    itk::Point<double, DIMENSION> datum;
    while (bstring_not_empty (line = bs.readLine ('\n'))) {
	float x, y;
	int rc = sscanf ((const char*) line, "%f %f", &x, &y);
	if (rc != 2) {
	    print_and_exit ("Error parsing file %s for read\n", filename);
	}
	datum[0] = x;
	datum[1] = y;
	data.push_back (datum);
    }
    fclose (fp);
}

int 
main (int argc, char *argv[])
{
    typedef itk::RANSAC< DoublePoint2DType, double> RANSACType;

    if (argc != 3) {
	printf ("Usage: ransac_test infile outfile\n");
	exit (0);
    }

    Point_vector data;
    load_data (data, argv[1]);
  
    std::vector<double> planeParameters;

    itk::Autolabel_ransac_est::Pointer planeEstimator 
	= itk::Autolabel_ransac_est::New();
    planeEstimator->SetDelta (0.5);

    //create and initialize the RANSAC algorithm
    double desiredProbabilityForNoOutliers = 0.999;
    double percentageOfDataUsed;
    RANSACType::Pointer ransacEstimator = RANSACType::New();
    ransacEstimator->SetData( data );
    ransacEstimator->SetParametersEstimator( planeEstimator.GetPointer() );
    percentageOfDataUsed = ransacEstimator->Compute (
	planeParameters, desiredProbabilityForNoOutliers);
    if (planeParameters.empty()) {
	std::cout<<"RANSAC estimate failed, degenerate configuration?\n";
    } else {
	printf ("RANSAC parameters: [s,i] = [%f,%f]\n", 
	    planeParameters[0], planeParameters[1]);
	printf ("Used %f percent of data.\n", percentageOfDataUsed);
    }

    // Dump output
    FILE *fp = fopen (argv[2], "w");

    Point_vector::iterator it;
    double slope = planeParameters[0];
    double intercept = planeParameters[1];
    for (it = data.begin(); it != data.end(); it++) {
	double x = (*it)[0];
	double y = (*it)[1];
	double ry = intercept + slope * x;
	fprintf (fp, "%f,%f,%f\n", x, y, ry);
    }
    std::cout << std::endl;

    fclose (fp);

    return EXIT_SUCCESS;
}
