#define ADD_MSG 0

class CLAddMessage : public CLMessage
{
public:
	CLAddMessage(int Op1, int Op2) : CLMessage(ADD_MSG)
	{
		m_Op1 = Op1;
		m_Op2 = Op2;
	}

	virtual ~CLAddMessage()
	{
	}

private:
	int m_Op1;
	int m_Op2;
};
