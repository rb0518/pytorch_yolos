#include "BaseLoss.h"

void BaseLoss::addNewItems(torch::Tensor loss, torch::Tensor lossitems)
{
	if (0 == m_times)
	{
		m_loss = torch::zeros_like(loss);
		m_lossitems = torch::zeros_like(lossitems);
	}

	m_loss = loss.clone();
	m_lossitems = m_lossitems + lossitems;
	m_times += 1;
}

void BaseLoss::reset()
{
	m_times = 0;
}
