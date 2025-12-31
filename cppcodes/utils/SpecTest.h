#pragma once
#include <string>
#include <memory>

#include "BaseModel.h"

void Test_TorchFlip_Funs();

void test_model(std::shared_ptr<BaseModelImpl> ptr_model, std::string strfilename = "../../data/images/bus.jpg");
