#include "plm_config.h"
#include <fstream>
#include "RANSAC.h"
#include "PlaneParametersEstimator.h"
#include "RandomNumberGenerator.h"
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

double
get_plane_y (
    std::vector<double>& plane,
    double x
)
{
    return - ((x - plane[2]) * plane[0]) / plane[1] + plane[3];
}

int 
main (int argc, char *argv[])
{
    typedef itk::PlaneParametersEstimator<DIMENSION> PlaneEstimatorType;
    typedef itk::RANSAC< DoublePoint2DType, double> RANSACType;

    if (argc != 3) {
	printf ("Usage: ransac_test infile outfile\n");
	exit (0);
    }

    Point_vector data;
    load_data (data, argv[1]);
  
    std::vector<double> planeParameters;
    unsigned int i;  

    //create and initialize the parameter estimator
    double maximalDistanceFromPlane = 0.5;
    PlaneEstimatorType::Pointer planeEstimator = PlaneEstimatorType::New();
    planeEstimator->SetDelta( maximalDistanceFromPlane );
    planeEstimator->LeastSquaresEstimate( data, planeParameters );
    if( planeParameters.empty() )
	std::cout<<"Least squares estimate failed, degenerate configuration?\n";
    else
    {
	std::cout<<"Least squares hyper(plane) parameters: [n,a]\n\t [ ";
	for( i=0; i<(2*DIMENSION-1); i++ )
	    std::cout<<planeParameters[i]<<", ";
	std::cout<<planeParameters[i]<<"]\n\n";
    }

    //create and initialize the RANSAC algorithm
    double desiredProbabilityForNoOutliers = 0.999;
    double percentageOfDataUsed;
    RANSACType::Pointer ransacEstimator = RANSACType::New();
    ransacEstimator->SetData( data );
    ransacEstimator->SetParametersEstimator( planeEstimator.GetPointer() );
    percentageOfDataUsed = 
	ransacEstimator->Compute( planeParameters, desiredProbabilityForNoOutliers );
    if( planeParameters.empty() )
	std::cout<<"RANSAC estimate failed, degenerate configuration?\n";
    else
    {
	std::cout<<"RANSAC hyper(plane) parameters: [n,a]\n\t [ ";
	for( i=0; i<(2*DIMENSION-1); i++ )
	    std::cout<<planeParameters[i]<<", ";
	std::cout<<planeParameters[i]<<"]\n\n";
	std::cout<<"\tPercentage of points which were used for final estimate: ";
	std::cout<<percentageOfDataUsed<<"\n\n";

    }

    // Dump output
    FILE *fp = fopen (argv[2], "w");

    Point_vector::iterator it;
    for (it = data.begin(); it != data.end(); it++) {
	double x = (*it)[0];
	double y = (*it)[1];

	double ry = get_plane_y (planeParameters, x);
	fprintf (fp, "%f,%f,%f\n", x, y, ry);
    }
    std::cout << std::endl;

    fclose (fp);

    return EXIT_SUCCESS;
}

