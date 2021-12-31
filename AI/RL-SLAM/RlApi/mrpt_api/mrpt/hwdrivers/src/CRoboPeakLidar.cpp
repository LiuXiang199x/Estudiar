/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2015, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "hwdrivers-precomp.h"   // Precompiled headers

#include <mrpt/hwdrivers/CRoboPeakLidar.h>
#include <mrpt/hwdrivers/CSerialPort.h>
#include <mrpt/utils/CClientTCPSocket.h>
#include <mrpt/system/os.h>

#define MRPT_HAS_ROBOPEAK_LIDAR   1

IMPLEMENTS_GENERIC_SENSOR(CRoboPeakLidar,mrpt::hwdrivers)

#if MRPT_HAS_ROBOPEAK_LIDAR
#	include "rplidar/include/rplidar.h" // RPLidar API
using namespace rp::standalone::rplidar;
#	define RPLIDAR_DRV   static_cast<RPlidarDriver*>(m_rplidar_drv)
#endif

using namespace mrpt::utils;
using namespace mrpt::slam;
using namespace mrpt::hwdrivers;
using namespace mrpt::system;
using namespace std;


/*-------------------------------------------------------------
						Constructor
-------------------------------------------------------------*/
CRoboPeakLidar::CRoboPeakLidar() :
	m_com_port("/dev/ttyS1"),
	m_com_port_baudrate(115200),
	m_rplidar_drv(NULL),
	m_adjust_angle(0)
{
	m_sensorLabel = "RPLidar";
}

/*-------------------------------------------------------------
					~CRoboPeakLidar
-------------------------------------------------------------*/
CRoboPeakLidar::~CRoboPeakLidar()
{
	turnOff();
	disconnect();
}

void CRoboPeakLidar::disconnect()
{
#if MRPT_HAS_ROBOPEAK_LIDAR
	RPlidarDriver::DisposeDriver(RPLIDAR_DRV);
	m_rplidar_drv=NULL;
#endif
}

/*-------------------------------------------------------------
						symmetryLidarData
-------------------------------------------------------------*/
void CRoboPeakLidar::symmetryLidarData(mrpt::slam::CObservation2DRangeScan	&outObservation)
{
    /* trans lidar data from clockwise to anti-clockwise */
    float scan[360]={0.0};
    char validRange[360]={false};
    for(int i = 0; i < (int)outObservation.scan.size(); i++)
    {
        if(i == 0)
        {
            scan[i] = outObservation.scan[i];
            validRange[i] = outObservation.validRange[i];
        }
        else
        {
            scan[360 - i] = outObservation.scan[i];
            validRange[360-i] = outObservation.validRange[i];
        }
    }

    for(int i = 0; i < (int)outObservation.scan.size(); i++)
    {
        outObservation.scan[i] = scan[i];
        outObservation.validRange[i] = validRange[i];
    }
}

/*-------------------------------------------------------------
						doProcessSimple
-------------------------------------------------------------*/
void  CRoboPeakLidar::doProcessSimple(
		bool							&outThereIsObservation,
		mrpt::slam::CObservation2DRangeScan	&outObservation,
		bool							&hardwareError )
{
#if MRPT_HAS_ROBOPEAK_LIDAR
	outThereIsObservation	= false;
	hardwareError			= false;

	// Bound?
	if (!checkCOMMs())
	{
		hardwareError = true;
		return;
	}

	rplidar_response_measurement_node_t nodes[360*2];
	size_t   count = sizeof(nodes)/sizeof(nodes[0]);

	// Scan:
	const mrpt::system::TTimeStamp tim_scan_start = mrpt::system::now();
	u_result op_result = RPLIDAR_DRV->grabScanData(nodes, count);
	//const mrpt::system::TTimeStamp tim_scan_end = mrpt::system::now();
	//const double scan_duration = mrpt::system::timeDifference(tim_scan_start,tim_scan_end);

    const size_t angle_compensate_nodes_count = 360;

	// Fill in scan data:
	if (op_result == RESULT_OK)
	{
		op_result = RPLIDAR_DRV->ascendScanData(nodes, count);
		if (op_result == RESULT_OK)
		{
            //-- cgli --
			const size_t angle_compensate_multiple = 1;
			int angle_compensate_offset = 0;
			rplidar_response_measurement_node_t angle_compensate_nodes[angle_compensate_nodes_count];
			memset(angle_compensate_nodes, 0, angle_compensate_nodes_count*sizeof(rplidar_response_measurement_node_t));

			//-- cgli --
			outObservation.scan.assign(angle_compensate_nodes_count, 0);
			outObservation.validRange.resize(angle_compensate_nodes_count, 0);

			for(size_t i=0 ; i < count; i++ )
			{
				if (nodes[i].distance_q2 != 0)
				{
					float angle = (float)((nodes[i].angle_q6_checkbit >> RPLIDAR_RESP_MEASUREMENT_ANGLE_SHIFT)/64.0f);
					int angle_value = (int)(angle * angle_compensate_multiple);
					if ((angle_value - angle_compensate_offset) < 0) angle_compensate_offset = angle_value;
					for (size_t j = 0; j < angle_compensate_multiple; j++)
					{
						angle_compensate_nodes[angle_value-angle_compensate_offset+j] = nodes[i];
					}
				}
			}

			//-- cgli --
			for(size_t i=0 ; i < angle_compensate_nodes_count; i++ )
			{
				const float read_value = (float) angle_compensate_nodes[i].distance_q2/4.0f/1000;
				outObservation.scan[i] = read_value;
				outObservation.validRange[i] = (read_value>0);
			}
		}
		else if (op_result == RESULT_OPERATION_FAIL)
		{
			// All the data is invalid, just publish them
			//-- cgli --
			outObservation.scan.assign(angle_compensate_nodes_count, 0);
			outObservation.validRange.resize(angle_compensate_nodes_count, 0);
		}
        vector<float> scan_array;
        vector<char> scan_valid;
        scan_array.resize(outObservation.scan.size());
        scan_valid.resize(outObservation.scan.size());
        if(m_adjust_angle > 0)
        {
            for(int i = 0; i < (int)outObservation.scan.size(); i++)
            {
                int index = i + m_adjust_angle;
                if(index >= (int)outObservation.scan.size())
                {
                    index = index - (int)outObservation.scan.size();
                }
                scan_array[index] = outObservation.scan[i];
                scan_valid[index] = outObservation.validRange[i];
            }
        }
        else
        {
            for(int i = 0; i < (int)outObservation.scan.size(); i++)
            {
                int index = i + m_adjust_angle;
                if(index < 0)
                {
                    index = (int)outObservation.scan.size() + index;
                }
                scan_array[index] = outObservation.scan[i];
                scan_valid[index] = outObservation.validRange[i];
            }
        }
        for(int i=0; i < (int)outObservation.scan.size(); i++)
        {
            outObservation.scan[i] = scan_array[i];
            outObservation.validRange[i] = scan_valid[i];
        }

        //symmetry lidar data
        symmetryLidarData(outObservation);

		// Fill common parts of the obs:
        outObservation.timestamp = tim_scan_start;
        outObservation.rightToLeft = true;
        outObservation.aperture = 2.0*M_PI;
        outObservation.maxRange	= 4.0;
        outObservation.stdError = 0.010f;
        outObservation.sensorPose = m_sensorPose;
        outObservation.sensorLabel = m_sensorLabel;

		// Do filter:
		C2DRangeFinderAbstract::filterByExclusionAreas( outObservation );
		C2DRangeFinderAbstract::filterByExclusionAngles( outObservation );
		// Do show preview:
		//C2DRangeFinderAbstract::processPreview(outObservation);

		outThereIsObservation = true;
	}
	else
	{
		if (op_result==RESULT_OPERATION_TIMEOUT || op_result==RESULT_OPERATION_FAIL)
		{
			// Error? Retry connection
			std::string erro = "RESULT_OPERATION_TIMEOUT";
			if(op_result==RESULT_OPERATION_FAIL)
			{
			    erro = "RESULT_OPERATION_FAIL";
			}

			std::cout << "[CRoboPeakLidar] lidar erro disconnect," << erro << std::endl;

//			this->disconnect();
		}
	}


#endif
}

/*-------------------------------------------------------------
						loadConfig_sensorSpecific
-------------------------------------------------------------*/
void  CRoboPeakLidar::loadConfig_sensorSpecific(
		const mrpt::utils::CConfigFileBase &configSource,
		const std::string	  &iniSection )
{
	m_sensorPose.setFromValues(
				configSource.read_float(iniSection,"pose_x",0),
				configSource.read_float(iniSection,"pose_y",0),
				configSource.read_float(iniSection,"pose_z",0),
				DEG2RAD( configSource.read_float(iniSection,"pose_yaw",0) ),
				DEG2RAD( configSource.read_float(iniSection,"pose_pitch",0) ),
				DEG2RAD( configSource.read_float(iniSection,"pose_roll",0) )
				);

#ifdef MRPT_OS_WINDOWS
	m_com_port = configSource.read_string(iniSection, "COM_port_WIN", m_com_port, true );
#else
	m_com_port = configSource.read_string(iniSection, "COM_port_LIN", m_com_port, true );
#endif

	// Parent options:
	this->loadExclusionAreas(configSource,iniSection);
}

/*-------------------------------------------------------------
						turnOn
-------------------------------------------------------------*/
bool  CRoboPeakLidar::turnOn()
{
	return checkCOMMs();
}

/*-------------------------------------------------------------
						turnOff
-------------------------------------------------------------*/
bool  CRoboPeakLidar::turnOff()
{
#if MRPT_HAS_ROBOPEAK_LIDAR
	if (RPLIDAR_DRV) RPLIDAR_DRV->stop();
	return true;
#else
	THROW_EXCEPTION("MRPT has been compiled without RPLidar support!")
		#endif
}

/** Returns true if the device is connected & operative */
bool CRoboPeakLidar::getDeviceHealth() const
{
#if MRPT_HAS_ROBOPEAK_LIDAR
	if (!RPLIDAR_DRV) return false;

	rplidar_response_device_health_t healthinfo;

	u_result op_result = RPLIDAR_DRV->getHealth(healthinfo);
	if (IS_OK(op_result))
	{
		printf("[CRoboPeakLidar] RPLidar health status : %d\n", healthinfo.status);
		if (healthinfo.status != RPLIDAR_STATUS_OK)
		{
			fprintf(stderr, "[CRoboPeakLidar] Error, rplidar internal error detected. Please reboot the device to retry.\n");
			return false;
		}
	}
	else
	{
		fprintf(stderr, "[CRoboPeakLidar] Error, cannot retrieve rplidar health code: %x\n", op_result);
		return false;
	}

	return true;
#else
	THROW_EXCEPTION("MRPT has been compiled without RPLidar support!")
		#endif
}



/*-------------------------------------------------------------
						checkCOMMs
-------------------------------------------------------------*/
bool  CRoboPeakLidar::checkCOMMs()
{
	MRPT_START
    #if MRPT_HAS_ROBOPEAK_LIDAR
        if (RPLIDAR_DRV)
        return true;

	// create the driver instance
	m_rplidar_drv = RPlidarDriver::CreateDriver(RPlidarDriver::DRIVER_TYPE_SERIALPORT);
	ASSERTMSG_(m_rplidar_drv, "Create Driver failed.")

    // Is it COMX, X>4? ->  "\\.\COMX"
    if (m_com_port.size()>=3)
	{
		if ( tolower( m_com_port[0]) =='c' && tolower( m_com_port[1]) =='o' && tolower( m_com_port[2]) =='m' )
		{
			// Need to add "\\.\"?
			if (m_com_port.size()>4 || m_com_port[3]>'4')
				m_com_port = std::string("\\\\.\\") + m_com_port;
		}
	}

	// make connection...
	if (IS_FAIL(RPLIDAR_DRV->connect( m_com_port.c_str(), (_u32)m_com_port_baudrate)))
	{
		fprintf(stderr, "[CRoboPeakLidar] Error, cannot bind to the specified serial port %s\n",  m_com_port.c_str() );
		return false;
	}

	rplidar_response_device_info_t devinfo;
	// change by Kimbo
	if (IS_FAIL(RPLIDAR_DRV->getDeviceInfo(devinfo) ))
	{
		return false;
	}

	if (m_verbose)
	{
		printf("[CRoboPeakLidar] Connection established:\n"
			   "Firmware version: %u\n"
			   "Hardware version: %u\n"
			   "Model: %u\n"
			   "Serial: ",
			   (unsigned int)devinfo.firmware_version,
			   (unsigned int)devinfo.hardware_version,
			   (unsigned int)devinfo.model);

		for (int i=0;i<16;i++)
			printf("%02X",devinfo.serialnum[i]);
		printf("\n");
	}

	// check health:
	if (!getDeviceHealth())
		return false;

	// start scan...
	u_result res = RPLIDAR_DRV->startScan();
	if (IS_FAIL( res ))
	{
		fprintf(stderr, "[CRoboPeakLidar] Error starting scanning mode: %x\n", res);
		return false;
	}

	return true;
#else
			THROW_EXCEPTION("MRPT has been compiled without RPLidar support!")
		#endif
			MRPT_END
}

/*-------------------------------------------------------------
						initialize
-------------------------------------------------------------*/
bool CRoboPeakLidar::initialize()
{
	if (!checkCOMMs()) return false;

	if (!turnOn())
	{
		cerr << "[CRoboPeakLidar::initialize] Error initializing RPLIDAR scanner" << endl;
		return false;
	}

	return true;

}


void CRoboPeakLidar::setSerialPort(const std::string &port_name)
{
	if (m_rplidar_drv)
		THROW_EXCEPTION("Can't change serial port while connected!")

				m_com_port = port_name;
}

