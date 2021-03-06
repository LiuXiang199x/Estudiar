/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */


#include <mrpt/utils/utils_defs.h>

#if defined(MRPT_OS_LINUX) || defined(MRPT_OS_APPLE)

// Refer to:
//  http://www.easysw.com/~mike/serial/serial.html

#include <stdio.h>   /* Standard input/output definitions */
#include <string.h>  /* String function definitions */
#include <unistd.h>  /* UNIX standard function definitions */
#include <fcntl.h>   /* File control definitions */
#include <errno.h>   /* Error number definitions */
#include <sys/time.h> // gettimeofday

#include <termios.h> /* POSIX terminal control definitions */

#include <sys/ioctl.h>  // FIONREAD,...
#include <signal.h>

#ifdef HAVE_LINUX_SERIAL_H
    #include <linux/serial.h>
#endif

#include <map>
#include <mrpt/system/threads.h>
#include <mrpt/hwdrivers/CSerialPort.h>

using namespace mrpt;
using namespace mrpt::hwdrivers;
using namespace std;

/* -----------------------------------------------------
                Constructor
   ----------------------------------------------------- */
CSerialPort::CSerialPort( const string &portName, bool openNow )
{
    hCOM = -1;	// Not connected

    m_totalTimeout_ms      = 0;
    m_interBytesTimeout_ms = 0;

    m_serialName = portName;

    if (openNow) open();
}

/* -----------------------------------------------------
                Default constructor
   ----------------------------------------------------- */
CSerialPort::CSerialPort()
{
    hCOM = -1;	// Not connected
    m_totalTimeout_ms      = 0;
    m_interBytesTimeout_ms = 0;
    m_serialName = "";
}


/* -----------------------------------------------------
                Destructor
   ----------------------------------------------------- */
CSerialPort::~CSerialPort()
{
    if ( isOpen() )
        close();
}

/* -----------------------------------------------------
                Open
   ----------------------------------------------------- */
void  CSerialPort::open( )
{
    MRPT_TRY_START

    // Check name:
    if (!m_serialName.size()) THROW_EXCEPTION("Serial port name is empty!!")
        if (m_serialName[0]!='/') m_serialName = string("/dev/") + m_serialName;

    // Open the serial port:
	// The O_NOCTTY flag tells UNIX that this program doesn't want to be the "controlling terminal" for that port.
	// The O_NDELAY flag tells UNIX that this program doesn't care what state the DCD signal line is in - whether the other end of the port is up and running.
    if ( -1==( hCOM= ::open( m_serialName.c_str(),  O_RDWR | O_NOCTTY | O_NDELAY ) ) )
        THROW_EXCEPTION_CUSTOM_MSG1("Error trying to open the serial port %s!!",m_serialName.c_str());

    // Clear flags:
    fcntl( hCOM, F_SETFL, 0 );

    //
    // Start assembling the new port settings.
    //
    termios port_settings;
    bzero( &port_settings,sizeof( port_settings ) ) ;

    //
    // Enable the receiver (CREAD) and ignore modem control lines
    // (CLOCAL).
    //
    port_settings.c_cflag |= CREAD | CLOCAL ;

    //
    // Set the VMIN and VTIME parameters to zero by default. VMIN is
    // the minimum number of characters for non-canonical read and
    // VTIME is the timeout in deciseconds for non-canonical
    // read. Setting both of these parameters to zero implies that a
    // read will return immediately only giving the currently
    // available characters.
    //
    port_settings.c_cc[ VMIN  ] = 0 ;
    port_settings.c_cc[ VTIME ] = 0 ;

    /*
     * Flush the input buffer associated with the port.
     */
    if ( tcflush( hCOM,TCIFLUSH ) < 0 )
		THROW_EXCEPTION_CUSTOM_MSG1("Cannot flush serial port: %s",strerror(errno) );

    /*
     * Write the new settings to the port.
     */
    if ( tcsetattr( hCOM,TCSANOW,&port_settings ) < 0 )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot set the new config to the serial port: %s",strerror(errno) ) ;


	// Do NOT block on read.
    //fcntl(hCOM, F_SETFL, FNDELAY);	//add by bruce
    fcntl(hCOM, F_SETFL, O_NDELAY);

    // Success!
    MRPT_TRY_END
}


/* -----------------------------------------------------
                isOpen
   ----------------------------------------------------- */
bool  CSerialPort::isOpen() const
{
    return hCOM != -1;
}

/* -----------------------------------------------------
                setConfig
   ----------------------------------------------------- */
void  CSerialPort::setConfig(
    int		baudRate,
    int		parity,
    int		bits,
    int		nStopBits,
    bool    enableFlowControl )
{
    MRPT_TRY_START

    // Port must be open!
    if (!isOpen()) THROW_EXCEPTION("The serial port is not open!");

    ASSERT_(baudRate>0)

    //
    // Apply baud rate
    //
    int BR;
    bool special_rate = false;
    switch (baudRate)
    {
    	case 50: BR = B50; break;
    	case 75: BR = B75; break;
    	case 110: BR = B110; break;
    	case 134: BR = B134; break;
    	case 200: BR = B200; break;
    	case 300: BR = B300; break;
    	case 600: BR = B600; break;
    	case 1200: BR = B1200; break;
    	case 2400: BR = B2400; break;
    	case 4800: BR = B4800; break;
    	case 9600: BR = B9600; break;
    	case 19200: BR = B19200; break;
    	case 38400: BR = B38400; break;
    	case 57600: BR = B57600; break;
    	case 115200: BR = B115200; break;
    	case 230400: BR = B230400; break;
#ifdef B460800
    	case 460800: BR = B460800; break;
#endif
#ifdef B500000
    	case 500000: BR = B500000; break;
#endif
#ifdef B4000000
    	case 576000: BR = B576000; break;
    	case 921600:  BR = B921600; break;
    	case 1000000:  BR = B1000000; break;
    	case 1152000:  BR = B1152000; break;
    	case 1500000:  BR = B1500000; break;
    	case 2000000:  BR = B2000000; break;
    	case 2500000:  BR = B2500000; break;
    	case 3000000:  BR = B3000000; break;
    	case 3500000:  BR = B3500000; break;
    	case 4000000:  BR = B4000000; break;
#endif
    default:
#ifdef HAVE_LINUX_SERIAL_H
        special_rate = true;
#else
        BR = baudRate; // This is all we can try in that case...
#endif
        break;
    }

    if (special_rate)
    {
#ifdef HAVE_LINUX_SERIAL_H
		struct serial_struct serial;
		if (ioctl(hCOM, TIOCGSERIAL, &serial) < 0)
            THROW_EXCEPTION("error on TIOCGSERIAL ioctl");

        serial.custom_divisor = serial.baud_base / baudRate;
        if (!serial.custom_divisor) serial.custom_divisor = 1;
        const int actual_rate = serial.baud_base / serial.custom_divisor;

		serial.flags &= ~ASYNC_SPD_MASK;
		serial.flags |= ASYNC_SPD_CUST; // We want to use our CUSTOM divisor.

		if (ioctl(hCOM, TIOCSSERIAL, &serial) < 0)
            THROW_EXCEPTION("error on TIOCSSERIAL ioctl");

        BR = B38400;  // It seems that 38400 means to the driver here to use our custom divisor

        if (actual_rate!=baudRate)
            cout << "[CSerialPort::setConfig] Setting custom baud rate to " << actual_rate << ", the closer I can make to " << baudRate << endl;
#else
        THROW_EXCEPTION("Custom serial port baud rates require linux/serial.h");
#endif
    } // end specialRate
    else
    {
        // Normal baudrate: Just in case, undo possible custom divisors:
//#ifdef HAVE_LINUX_SERIAL_H
//		struct serial_struct serial;
//		if (ioctl(hCOM, TIOCGSERIAL, &serial) < 0)
//            THROW_EXCEPTION("error on TIOCGSERIAL ioctl");
//
//        serial.flags &= ~ASYNC_SPD_MASK;
//
//		if (ioctl(hCOM, TIOCSSERIAL, &serial) < 0)
//            THROW_EXCEPTION("error on TIOCSSERIAL ioctl");
//#endif
    }

    termios port_settings;
    if ( tcgetattr( hCOM, & port_settings ) < 0 )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot get the current settings: %s",strerror(errno) ) ;

    if ( ( cfsetispeed( &port_settings,BR ) < 0 ) ||
            ( cfsetospeed( &port_settings,BR) < 0 ) )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot change baudRate in setting structure: %s",strerror(errno) ) ;

    //
    // Set the character size.
    //
    port_settings.c_cflag &= ~CSIZE ;
    switch (bits)
    {
    case 5:
        port_settings.c_cflag |= CS5;
        break;
    case 6:
        port_settings.c_cflag |= CS6;
        break;
    case 7:
        port_settings.c_cflag |= CS7;
        break;
    case 8:
        port_settings.c_cflag |= CS8;
        break;
    default:
        THROW_EXCEPTION_CUSTOM_MSG1("Invalid character size: %i",bits ) ;
        break;
    }

    // parity  0:No parity, 1:Odd, 2:Even
    switch ( parity )
    {
    case 2:
        port_settings.c_cflag |= PARENB ;
        port_settings.c_cflag &= ~PARODD ;
        port_settings.c_iflag |= INPCK ;
        break ;
    case 1:
        port_settings.c_cflag |= ( PARENB | PARODD );
        port_settings.c_iflag |= INPCK;
        break ;
    case 0:
        port_settings.c_cflag &= ~(PARENB);
        port_settings.c_iflag |= IGNPAR;
        break ;
    default:
        THROW_EXCEPTION_CUSTOM_MSG1("Invalid parity selection: %i",parity) ;
        break;
    }

    // stop bits:
    switch ( nStopBits )
    {
    case 1:
        port_settings.c_cflag &= ~(CSTOPB) ;
        break ;
    case 2:
        port_settings.c_cflag |= CSTOPB ;
        break ;
    default:
        THROW_EXCEPTION_CUSTOM_MSG1("Invalid number of stop bits: %i",nStopBits) ;
        break;
    }

    //
    // Set the flow control.
    //
    if (enableFlowControl)
    {
        // RTS/CTS ON:
        port_settings.c_cflag |= CRTSCTS ;
    }
    else
    {
        // none
        port_settings.c_cflag &= ~(CRTSCTS) ;
    }

    /* Write the new settings to the port.
     */
    if ( tcsetattr( hCOM,TCSANOW,&port_settings ) < 0 )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot set the new settings: %s",strerror(errno) );

    // Check:
    termios port_settings_verif;
    if ( tcgetattr( hCOM, & port_settings_verif) < 0 )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot get the settings to verify: %s",strerror(errno) ) ;

#if 0
    if (!special_rate)
    {
        if (port_settings_verif.c_ispeed != port_settings.c_ispeed)
            THROW_EXCEPTION("Verification of changed baudrate(i) failed");
        if (port_settings_verif.c_ospeed != port_settings.c_ospeed)
            THROW_EXCEPTION("Verification of changed baudrate(i) failed");
    }

    if (port_settings_verif.c_cflag !=  port_settings.c_cflag)
        THROW_EXCEPTION("Verification of serial port flags failed");
#endif

    m_baudRate = baudRate;

    MRPT_TRY_END
}

/* -----------------------------------------------------
                setConfig
   ----------------------------------------------------- */
void  CSerialPort::setTimeouts(
    int		ReadIntervalTimeout,
    int		ReadTotalTimeoutMultiplier,
    int		ReadTotalTimeoutConstant,
    int		WriteTotalTimeoutMultiplier,
    int		WriteTotalTimeoutConstant )
{
    MRPT_TRY_START

    // Port must be open!
    if (!isOpen()) THROW_EXCEPTION("The serial port is not open!");


    // Save variables which are used in other methods:
    m_totalTimeout_ms      = ReadTotalTimeoutConstant;
    m_interBytesTimeout_ms = ReadIntervalTimeout;


    // http://www.unixwiz.net/techtips/termios-vmin-vtime.html
    // VMIN & VTIME
    termios port_settings;
    if ( tcgetattr( hCOM, & port_settings ) < 0 )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot get the current settings: %s",strerror(errno) ) ;

    //
    // We set VMIN=0 and VTIME=ReadIntervalTimeout (in thenth of seconds)
    //
    port_settings.c_cc[ VMIN  ] = 0;
    port_settings.c_cc[ VTIME ] = max(1,ReadTotalTimeoutConstant / 100);

    /* Write the new settings to the port.
     */
    if ( tcsetattr( hCOM,TCSANOW,&port_settings ) < 0 )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot set the new settings: %s",strerror(errno) ) ;

    // Success
    MRPT_TRY_END
}

/* -----------------------------------------------------
                Close
   ----------------------------------------------------- */
void  CSerialPort::close(  )
{
    MRPT_TRY_START

    if (hCOM<0) return; // Already closed

    //
//    PosixSignalDispatcher& signal_dispatcher = PosixSignalDispatcher::Instance() ;
//    signal_dispatcher.DetachHandler( SIGIO, *this ) ;

    // Close the serial port file descriptor.
    ::close(hCOM);

    hCOM=-1;	// Means the port is closed

    MRPT_TRY_END
}

/* -----------------------------------------------------
                read
   ----------------------------------------------------- */
size_t  CSerialPort::Read(void *Buffer, size_t Count)
{
    MRPT_TRY_START

 //   cout << "READ: "<< Count << endl;

    // Port must be open!
    if (!isOpen()) THROW_EXCEPTION("The port is not open yet!");

	if (!Count) return 0;

    // Use the "m_totalTimeout_ms" global timeout
    //  and the "m_interBytesTimeout_ms" for inter-bytes:
    m_timer.Tic();

    size_t  alreadyRead = 0;
    int		leftTime = m_totalTimeout_ms - (int)(m_timer.Tac()*1000);

    while ( alreadyRead<Count && leftTime>=0 )
    {
    	// Bytes waiting in the queue?
		// Check if we are still connected or there is an error...
		int waiting_bytes=0;
		if ( ioctl(hCOM, FIONREAD, &waiting_bytes) < 0)
		{
			if (errno==EIO)
			{
				// The port has been disconnect (for USB ports)
				this->close();
				return alreadyRead;
			}
		}

		// Are there any bytes??
		int nRead=0;

		if (waiting_bytes>0)
		{
			int nToRead = min( (size_t)waiting_bytes, Count-alreadyRead );

			if ( ( nRead=::read(hCOM, ((char*)Buffer)+alreadyRead, nToRead ) ) <0 )
			{
				cerr << "[CSerialPort] Error reading from port..." << endl;
			}

			alreadyRead+= nRead;
		}
		else
		{
			// Nope...
		}

		if (alreadyRead<Count)
		{
			// Wait 1 more ms for new data to arrive.
			mrpt::system::sleep( 1 );
		}

		// Reset interbytes timer:
		leftTime = m_totalTimeout_ms - (int)(m_timer.Tac()*1000);
		if (nRead>0)
			leftTime = max(leftTime, m_interBytesTimeout_ms);
    }

//    cout << "READ DONE: "<< alreadyRead << endl;
    return alreadyRead;

    MRPT_TRY_END
}

/** Reads one text line from the serial port in POSIX "canonical mode".
  *  This method reads from the serial port until one of the characters in \a eol are found.
  */
std::string CSerialPort::ReadString(
	const int total_timeout_ms,
	bool *out_timeout,
	const char *eol_chars)
{
    MRPT_TRY_START
    // Calling ::ReadBuffer() many times would be even worse, so replicate its code here:

    ASSERT_(eol_chars!=NULL)

    // Port must be open!
    if (!isOpen()) THROW_EXCEPTION("The port is not open yet!");

    if (out_timeout) *out_timeout = false; // Will be set to true on timeout

    m_timer.Tic();
    string receivedStr; // Rx buffer

    while ( total_timeout_ms<0 || ( m_timer.Tac()*1e3 < total_timeout_ms ) )
    {
    	// Bytes waiting in the queue?
		// Check if we are still connected or there is an error...
		int waiting_bytes=0;
		if ( ioctl(hCOM, FIONREAD, &waiting_bytes) < 0)
		{
			if (errno==EIO)
			{	// The port has been disconnect (for USB ports)
				this->close();
				THROW_EXCEPTION("Error reading port before end of line")
			}
		}

		// Are there any bytes??
		int nRead=0;
		if (waiting_bytes>0)
		{
			// Read just 1 byte:
			char buf[1];
			if ( ( nRead=::read(hCOM, buf, 1) ) <0 )
			{
				cerr << "[CSerialPort] Error reading from port..." << endl;
			}
			if (nRead)
			{	// Append to string, if it's not a control char:
				if (!strchr(eol_chars, buf[0] ) )
					receivedStr.push_back( buf[0] );
				else
				{	// end of string!
					return receivedStr;
				}
			}
		}
		else
		{
			// we decide to move the sleep here to satisfy realtime requirement in the case where we are waiting a n-length string at a frequency
			// greater than 1/n...
			mrpt::system::sleep( 1 ); // Wait 1 more ms for new data to arrive.
		}
		// If we are still here, string is not finished:
    }

	// Timeout:
    if (out_timeout) *out_timeout = true;
    return receivedStr;
    MRPT_TRY_END
}


/* -----------------------------------------------------
                write
   ----------------------------------------------------- */
size_t  CSerialPort::Write(const void *Buffer, size_t Count)
{
    MRPT_TRY_START
    // Port must be open!
    if (!isOpen()) THROW_EXCEPTION("The port is not open yet!");

    // Write the data to the serial port. Keep retrying if EAGAIN
    // error is received.

    /** \todo Add support for write timeout here
      */
    struct timeval start, end;
    int num_of_bytes_written = -1 ;
    size_t total_bytes_written = 0;

    do
    {
        gettimeofday(&start, NULL);
        num_of_bytes_written = write( hCOM,reinterpret_cast<const char*>(Buffer)+total_bytes_written, Count-total_bytes_written );
        //cout << "wr: " << num_of_bytes_written << " tot: " << total_bytes_written << " of " << Count << " err: " << errno << endl;
        if (num_of_bytes_written>0)
            total_bytes_written+=num_of_bytes_written;

        if (num_of_bytes_written<(int)Count)
        {
            // JL: These few lines are from the Player/Stage project:

            // need to do this sort of busy wait to ensure the right timing
            // although I've noticed you will get some anamolies that are
            // in the ms range; this could be a problem...
            int usecs;
            do {
            gettimeofday(&end, NULL);
            usecs= (end.tv_sec - start.tv_sec)*1000000 +
              (end.tv_usec - start.tv_usec);
            } while (usecs < 60);
            //mrpt::system::sleep(1); // we'll continue writting is a ms.
        }
    }
    while ( ( total_bytes_written<Count) &&
            ( !errno || EAGAIN == errno ) ) ;
    //
    if ( num_of_bytes_written < 0 )  // This means we exit the loop due to a bad "errno".
        THROW_EXCEPTION_CUSTOM_MSG1("Error writing data to the serial port: %s",strerror(errno) ) ;

    // Make sure the queue is drained
    // Synchronous IO doesnt always work
    //::tcdrain(hCOM);	//add by bruce

    // OK:
    return total_bytes_written;

    MRPT_TRY_END
}

/* -----------------------------------------------------
                purgeBuffers
   ----------------------------------------------------- */
void  CSerialPort::purgeBuffers()
{
    MRPT_TRY_START

    // Port must be open!
    if (!isOpen()) THROW_EXCEPTION("The port is not open yet!");

    /*
     * Flush the input buffer associated with the port.
     */
    if ( tcflush( hCOM,TCIFLUSH ) < 0 )
        THROW_EXCEPTION_CUSTOM_MSG1("Cannot flush serial port: %s",strerror(errno) ) ;

    MRPT_TRY_END
}



#endif  // linux

