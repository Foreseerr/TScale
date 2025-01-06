#pragma once

namespace NHPTimer
{
	typedef i64 STime;
	double GetSeconds( const STime &a );
	// �������� ������� �����
	void GetTime( STime *pTime );
	// �������� �����, ��������� � �������, ����������� � *pTime, ��� ���� � *pTime ����� �������� ������� �����
	double GetTimePassed( STime *pTime );
	// �������� ������� ����������
	double GetClockRate();
	// recalc CPU frequency, call regularly to support SpeedStep processors
	void UpdateHPTimerFrequency();
};
