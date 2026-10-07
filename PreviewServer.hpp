#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <thread>

#include "logger.hpp"

/**
 * @brief 只为预览服务的最小 HTTP 服务器：一个线程，逐个处理 GET
 * 请求，每次响应后关闭连接。 Minimal HTTP server for the preview: one thread serves GET
 * requests one at a time and closes the connection after each response.
 */
class PreviewServer
{
 public:
  /// 一次响应 / One response.
  struct Response
  {
    int status = 200;
    std::string content_type = "text/plain";
    std::string body;
  };
  /// 按路径生成响应 / Builds the response for a path.
  using Handler = std::function<Response(std::string_view path)>;

  /**
   * @param port 监听端口，0 为由系统分配 / Listening port; 0 lets the system choose
   * @param handler 在服务线程里调用 / Called on the server thread
   */
  PreviewServer(uint16_t port, Handler handler) : handler_(std::move(handler))
  {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (listen_fd_ < 0 ||
        ::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(listen_fd_, 4) != 0)
    {
      XR_LOG_ERROR("preview: cannot listen on port %u", static_cast<unsigned>(port));
      Close(listen_fd_);
      return;
    }
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    running_.store(true);
    thread_ = std::thread([this]() { Loop(); });
  }

  ~PreviewServer()
  {
    running_.store(false);
    if (thread_.joinable())
    {
      thread_.join();
    }
    Close(listen_fd_);
  }

  PreviewServer(const PreviewServer&) = delete;
  PreviewServer& operator=(const PreviewServer&) = delete;

  /// 实际监听的端口，未能监听时为 0 / Actual port, 0 when listening failed.
  uint16_t Port() const { return port_; }

 private:
  static void Close(int& fd)
  {
    if (fd >= 0)
    {
      ::close(fd);
      fd = -1;
    }
  }

  void Loop()
  {
    while (running_.load())
    {
      pollfd p{listen_fd_, POLLIN, 0};
      if (::poll(&p, 1, 200) <= 0)
      {
        continue;
      }
      int client = ::accept(listen_fd_, nullptr, nullptr);
      if (client >= 0)
      {
        Serve(client);
        Close(client);
      }
    }
  }

  void Serve(int client)
  {
    // 只读请求头，取第一行的路径 / Read the header and take the path of the first line.
    std::string request;
    char buf[1024];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192)
    {
      pollfd p{client, POLLIN, 0};
      if (::poll(&p, 1, 1000) <= 0)
      {
        return;
      }
      const ssize_t n = ::recv(client, buf, sizeof(buf), 0);
      if (n <= 0)
      {
        return;
      }
      request.append(buf, static_cast<std::size_t>(n));
    }
    const std::size_t sp1 = request.find(' ');
    const std::size_t sp2 = request.find(' ', sp1 + 1);
    if (request.compare(0, 4, "GET ") != 0 || sp2 == std::string::npos)
    {
      Send(client, {405, "text/plain", "GET only\n"});
      return;
    }
    std::string_view path(request.data() + sp1 + 1, sp2 - sp1 - 1);
    path = path.substr(0, path.find('?'));
    Send(client, handler_(path));
  }

  static void Send(int client, const Response& r)
  {
    const char* reason = r.status == 200 ? "OK" : r.status == 404 ? "Not Found" : "Error";
    const std::string head = "HTTP/1.1 " + std::to_string(r.status) + " " + reason +
                             "\r\nContent-Type: " + r.content_type +
                             "\r\nContent-Length: " + std::to_string(r.body.size()) +
                             "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    if (WriteAll(client, head))
    {
      WriteAll(client, r.body);
    }
  }

  static bool WriteAll(int client, std::string_view data)
  {
    while (!data.empty())
    {
      const ssize_t n = ::send(client, data.data(), data.size(), MSG_NOSIGNAL);
      if (n <= 0)
      {
        return false;
      }
      data.remove_prefix(static_cast<std::size_t>(n));
    }
    return true;
  }

  Handler handler_;
  int listen_fd_ = -1;
  uint16_t port_ = 0;
  std::atomic<bool> running_{false};
  std::thread thread_;
};
