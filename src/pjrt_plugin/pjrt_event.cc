// PJRT Event API implementation for Metal backend.
//
// Events are backed by MLX stream events (see PJRT_Event in pjrt_types.h). The
// implementation here only ever touches the snapshotted mlx::core::Event
// copies, which wrap thread-safe MTLSharedEvents — it never reads or mutates an
// mlx::core::array, so the completion machinery is safe to run on a background
// thread concurrently with the main dispatch thread.

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/pjrt_types.h"

namespace {

// A single shared background thread that waits on MLX events and then fires
// PJRT OnReady callbacks. GPU work on a stream completes in submission order,
// so serial processing of the queue is both correct and efficient. The service
// is an intentionally-leaked singleton: the worker loops until process exit,
// which sidesteps any static-destruction-order hazards with Metal teardown.
class CompletionService {
public:
    CompletionService() : worker_([this] { Run(); }) {}

    void Enqueue(std::vector<mlx::core::Event> events, PJRT_Event_OnReadyCallback callback,
                 void* user_arg) {
        {
            std::scoped_lock lk(mu_);
            queue_.push_back({std::move(events), callback, user_arg});
        }
        cv_.notify_one();
    }

private:
    struct Entry {
        std::vector<mlx::core::Event> events;
        PJRT_Event_OnReadyCallback callback;
        void* user_arg;
    };

    void Run() {
        for (;;) {
            Entry entry;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return !queue_.empty(); });
                entry = std::move(queue_.front());
                queue_.pop_front();
            }

            PJRT_Error* error = nullptr;
            try {
                for (auto& event : entry.events) {
                    if (event.valid()) {
                        event.wait();
                    }
                }
            } catch (const std::exception& e) {
                error = new PJRT_Error{std::string("async completion: ") + e.what(),
                                       PJRT_Error_Code_INTERNAL};
            }
            // PJRT contract: the callback takes ownership of `error`.
            entry.callback(error, entry.user_arg);
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Entry> queue_;
    std::thread worker_;
};

CompletionService& GetCompletionService() {
    static CompletionService* service = new CompletionService();  // intentionally leaked
    return *service;
}

}  // namespace

// ============================================================================
// PJRT_Event method implementations
// ============================================================================

PJRT_Event::PJRT_Event(std::vector<mlx::core::array> arrays) {
    for (auto& array : arrays) {
        const mlx::core::Event& event = array.event();
        if (event.valid()) {
            events_.push_back(event);
        }
    }
    ready_ = events_.empty();
}

bool PJRT_Event::IsReady() {
    std::scoped_lock lk(mu_);
    if (ready_) {
        return true;
    }
    for (auto& event : events_) {
        if (event.valid() && !event.is_signaled()) {
            return false;
        }
    }
    ready_ = true;
    return true;
}

void PJRT_Event::Await() {
    std::vector<mlx::core::Event> snapshot;
    {
        std::scoped_lock lk(mu_);
        if (ready_) {
            return;
        }
        snapshot = events_;  // cheap: shared handles
    }
    // Wait outside the lock so concurrent IsReady()/OnReady() are not blocked.
    for (auto& event : snapshot) {
        if (event.valid()) {
            event.wait();
        }
    }
    std::scoped_lock lk(mu_);
    ready_ = true;
}

void PJRT_Event::OnReady(PJRT_Event_OnReadyCallback callback, void* user_arg) {
    if (!callback) {
        return;
    }
    std::vector<mlx::core::Event> snapshot;
    {
        std::scoped_lock lk(mu_);
        if (!ready_) {
            snapshot = events_;
        }
    }
    if (snapshot.empty()) {
        callback(nullptr, user_arg);
        return;
    }
    GetCompletionService().Enqueue(std::move(snapshot), callback, user_arg);
}

// ============================================================================
// Event C API
// ============================================================================

PJRT_Error* MPS_Event_Destroy(PJRT_Event_Destroy_Args* args) {
    delete args->event;
    return nullptr;
}

PJRT_Error* MPS_Event_IsReady(PJRT_Event_IsReady_Args* args) {
    args->is_ready = args->event ? args->event->IsReady() : true;
    return nullptr;
}

PJRT_Error* MPS_Event_Error(PJRT_Event_Error_Args* args) {
    // Runtime GPU-execution errors are not currently captured here (MLX surfaces
    // most failures synchronously during graph building). Returns no error.
    return nullptr;
}

PJRT_Error* MPS_Event_Await(PJRT_Event_Await_Args* args) {
    if (args->event) {
        // mlx::core::Event::wait() can throw; an exception must not cross the
        // PJRT C API boundary (UB). Surface it as a PJRT_Error instead.
        try {
            args->event->Await();
        } catch (const std::exception& e) {
            return new PJRT_Error{std::string("event await: ") + e.what(),
                                  PJRT_Error_Code_INTERNAL};
        }
    }
    return nullptr;
}

PJRT_Error* MPS_Event_OnReady(PJRT_Event_OnReady_Args* args) {
    if (args->event) {
        args->event->OnReady(args->callback, args->user_arg);
    } else if (args->callback) {
        args->callback(nullptr, args->user_arg);
    }
    return nullptr;
}
