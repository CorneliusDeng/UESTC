#ifndef CLSTATUS_H
#define CLSTATUS_H

/*
用于保存函数的处理结果
*/
class CLStatus
{
public:
	/*
	lReturnCode >=0表示成功，否则失败
	*/
	CLStatus(long lReturnCode, long lErrorCode);
	virtual ~CLStatus();

public:
	bool IsSuccess();
	long GetErrorCode();

private:
	long m_lReturnCode;
	long m_lErrorCode;
};

#endif
