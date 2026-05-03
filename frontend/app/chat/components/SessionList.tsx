/*
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: /app/chat/components/SessionList.tsx                                  │
│ Developed by: Davidson Gomes                                                 │
│ Creation date: May 13, 2025                                                  │
│ Contact: contato@evolution-api.com                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ @copyright © Evolution API 2025. All rights reserved.                        │
│ Licensed under the Apache License, Version 2.0                               │
│                                                                              │
│ You may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│    http://www.apache.org/licenses/LICENSE-2.0                                │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ @important                                                                   │
│ For any future changes to the code in this file, it is recommended to        │
│ include, together with the modification, the information of the developer    │
│ who changed it and the date of modification.                                 │
└──────────────────────────────────────────────────────────────────────────────┘
*/
"use client";

import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Search, Filter, Plus, Loader2 } from "lucide-react";
import { ChatSession } from "@/services/sessionService";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface SessionListProps {
  sessions: ChatSession[];
  agents: any[];
  selectedSession: string | null;
  isLoading: boolean;
  searchTerm: string;
  selectedAgentFilter: string;
  showAgentFilter: boolean;
  setSearchTerm: (value: string) => void;
  setSelectedAgentFilter: (value: string) => void;
  setShowAgentFilter: (value: boolean) => void;
  setSelectedSession: (value: string | null) => void;
  setIsNewChatDialogOpen: (value: boolean) => void;
}

export function SessionList({
  sessions,
  agents,
  selectedSession,
  isLoading,
  searchTerm,
  selectedAgentFilter,
  showAgentFilter,
  setSearchTerm,
  setSelectedAgentFilter,
  setShowAgentFilter,
  setSelectedSession,
  setIsNewChatDialogOpen,
}: SessionListProps) {
  const filteredSessions = sessions.filter((session) => {
    const matchesSearchTerm = session.id
      .toLowerCase()
      .includes(searchTerm.toLowerCase());

    if (selectedAgentFilter === "all") {
      return matchesSearchTerm;
    }

    const sessionAgentId = session.id.split("_")[1];
    return matchesSearchTerm && sessionAgentId === selectedAgentFilter;
  });

  const sortedSessions = [...filteredSessions].sort((a, b) => {
    const updateTimeA = new Date(a.update_time).getTime();
    const updateTimeB = new Date(b.update_time).getTime();

    return updateTimeB - updateTimeA;
  });

  const formatDateTime = (dateTimeStr: string) => {
    try {
      const date = new Date(dateTimeStr);

      const day = date.getDate().toString().padStart(2, "0");
      const month = (date.getMonth() + 1).toString().padStart(2, "0");
      const year = date.getFullYear();
      const hours = date.getHours().toString().padStart(2, "0");
      const minutes = date.getMinutes().toString().padStart(2, "0");

      return `${day}/${month}/${year} ${hours}:${minutes}`;
    } catch (error) {
      return "Invalid date";
    }
  };

  const getExternalId = (sessionId: string) => {
    return sessionId.split("_")[0];
  };

  return (
    <div className="w-64 border-r border-neutral-700 flex flex-col bg-neutral-900 h-full">
      <div className="p-4 border-b border-neutral-700">
        <div className="flex items-center justify-between mb-4">
          <Button
            onClick={() => setIsNewChatDialogOpen(true)}
            className="bg-emerald-800 text-emerald-100 hover:bg-emerald-700 border-emerald-700"
            size="sm"
          >
            <Plus className="h-4 w-4 mr-1" /> New Conversation
          </Button>
        </div>

        <div className="space-y-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-neutral-500" />
            <Input
              placeholder="Search conversations..."
              className="pl-9 bg-neutral-800 border-neutral-700 text-neutral-200 focus-visible:ring-emerald-500"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              size="sm"
              className="text-neutral-400 hover:text-white hover:bg-neutral-800"
              onClick={() => setShowAgentFilter(!showAgentFilter)}
            >
              <Filter className="h-4 w-4 mr-1" />
              Filter
            </Button>

            {selectedAgentFilter !== "all" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedAgentFilter("all")}
                className="text-neutral-400 hover:text-white hover:bg-neutral-800"
              >
                Clear filter
              </Button>
            )}
          </div>

          {showAgentFilter && (
            <div className="pt-1">
              <Select
                value={selectedAgentFilter}
                onValueChange={setSelectedAgentFilter}
              >
                <SelectTrigger className="bg-neutral-800 border-neutral-700 text-neutral-200">
                  <SelectValue placeholder="Filter by agent" />
                </SelectTrigger>
                <SelectContent className="bg-neutral-900 border-neutral-700 text-white">
                  <SelectItem
                    value="all"
                    className="data-[selected]:bg-neutral-800 data-[highlighted]:bg-neutral-800 !text-white hover:text-emerald-400 data-[selected]:!text-emerald-400"
                  >
                    All agents
                  </SelectItem>
                  {agents.map((agent) => (
                    <SelectItem
                      key={agent.id}
                      value={agent.id}
                      className="data-[selected]:bg-neutral-800 data-[highlighted]:bg-neutral-800 !text-white hover:text-emerald-400 data-[selected]:!text-emerald-400"
                    >
                      {agent.name.slice(0, 15)}{" "}
                      {agent.name.length > 15 && "..."}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
        {isLoading ? (
          <div className="flex justify-center items-center h-24">
            <Loader2 className="h-5 w-5 text-emerald-400 animate-spin" />
          </div>
        ) : sortedSessions.length > 0 ? (
          <div className="px-4 pt-2 space-y-2">
            {sortedSessions.map((session) => {
              const agentId = session.id.split("_")[1];
              const agentInfo = agents.find((a) => a.id === agentId);
              const externalId = getExternalId(session.id);

              return (
                <div
                  key={session.id}
                  className={`p-3 rounded-md cursor-pointer transition-colors group relative ${
                    selectedSession === session.id
                      ? "bg-emerald-800/20 border border-emerald-600/40"
                      : "bg-neutral-800 hover:bg-neutral-700 border border-transparent"
                  }`}
                  onClick={() => setSelectedSession(session.id)}
                >
                  <div className="flex items-center">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 mr-2"></div>
                    <div className="text-neutral-200 font-medium truncate max-w-[180px]">
                      {externalId}
                    </div>
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    {agentInfo && (
                      <Badge className="bg-neutral-700 text-emerald-400 border-neutral-600 text-xs">
                        {agentInfo.name.slice(0, 15)}
                        {agentInfo.name.length > 15 && "..."}
                      </Badge>
                    )}
                    <div className="text-xs text-neutral-500 ml-auto">
                      {formatDateTime(session.update_time)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : searchTerm || selectedAgentFilter !== "all" ? (
          <div className="text-center py-4 text-neutral-400">
            No results found
          </div>
        ) : (
          <div className="text-center py-4 text-neutral-400">
            Click "New" to start
          </div>
        )}
      </div>
    </div>
  );
}
