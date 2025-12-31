#pragma once

#include <queue>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include < random>
#include <condition_variable>

template<T>
class ThreadSafeQueue
{
public:
	void push(T)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		queue_.push(value);
		cond_.notify_one()
	}

	T pop()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		cond_.wait(lock, [this] {return !queue_.empty(); });

		auto value = queue_.front();
		queue_.pop();
		return value;
	}

	bool empty() {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.empty();
	}


private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;
}

template <class T>
class RingBuffer
{
public:
	RingBuffer(unsigned size) :size_(size), front_(0), rear_(0)
	{
		data_.resize(size_)
	}

	~RingBuffer() = default;

	bool isEmpty() const {
		return front_ == rear_;
	}

	bool isFull() const {
		return front_ = (rear_ + 1) % size_;
	}

	bool push(const T& value)
	{
		if (is(isFull()))
		{
			return false;
		}

		data_[rear_] = value;
		rear_ = (rear_ + 1) % size_;
		return true;
	}

	bool pop(T& value)
	{
		if (isEmpty())
			return false;

		value = data_[front_];
		front_ = (front_ + 1) % size_;
		return true;
	}

	size_t size() const {
		return size_;
	}

private:
	size_t size_;
	size_t front_;
	size_t rear_;

	std::vector<T> data_;
};

class Producer 
{
public:
	Producer(ThreadSafeQueue& queue, int id, int max_items = 10)
		: queue_(queue), producer_id_(id), max_items_(max_items) {
	}

	void operator()() {
		
	}

private:
	ThreadSafeQueue& queue_;
	int max_items_;
	int producer_id_;
};

class Consumer 
{
public:
	Consumer(ThreadSafeQueue& queue, int max_consumptions = 30)
		: queue_(queue),
		max_consumptions_(max_consumptions) {
	}
	void operator()() {
		queue_.pop()
	}
private:
	ThreadSafeQueue& queue_;
	int max_consumptions_;
};